"""Bounded JSONL client for a local SAM interpreter; never imports SAM itself."""
from collections import deque
from dataclasses import asdict
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import uuid

from .regions import mask_bbox, mask_digest, read_mask, resolve_asset
from .sam_config import SamConfig, finite_number

PROTOCOL_VERSION = 1


class SamWorkerError(RuntimeError):
    """A runtime failure, distinct from a successful empty segmentation."""


class SamWorker:
    def __init__(self, python, config: SamConfig, timeout=300.0, worker_path=None, cancelled=None):
        self.timeout = timeout
        self.cancelled = cancelled or (lambda: False)
        self.provenance = {}
        self.last_metrics = {}
        self.process = None
        self.lines = queue.Queue(maxsize=8)
        self.logs = deque(maxlen=30)
        self.closing = threading.Event()
        self.config = config
        worker_path = worker_path or Path(__file__).with_name("sam3_worker.py")
        command = [str(python), "-u", str(worker_path)]
        try:
            self.process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            self.reader = threading.Thread(target=self._read, args=(self.process.stdout,), daemon=True)
            self.stderr_reader = threading.Thread(target=self._read_errors, args=(self.process.stderr,), daemon=True)
            self.reader.start()
            self.stderr_reader.start()
            self.provenance = self.request("init", model={
                "checkpoint": config.checkpoint, "device": config.device,
                "allow_download": config.allow_download,
                "backend": config.backend, "model_id": config.model_id, "revision": config.revision,
            })["provenance"]
        except BaseException:
            self.close()
            raise

    def load(self):
        """Restore the initialized model on its configured device."""
        self.request("load")

    def offload(self):
        """Wait until SAM has released its GPU model before starting WD."""
        self.request("offload")

    def _read(self, stream):
        try:
            while not self.closing.is_set():
                line = stream.readline(4 * 1024 * 1024)
                while not self.closing.is_set():
                    try:
                        self.lines.put(line or None, timeout=.1)
                        break
                    except queue.Full:
                        continue
                if not line:
                    return
        except (OSError, ValueError):
            return

    def _read_errors(self, stream):
        try:
            for line in stream:
                self.logs.append(line[-2000:].rstrip())
        except (OSError, ValueError):
            pass

    def request(self, action, **payload):
        if self.process is None or self.process.poll() is not None:
            raise SamWorkerError("SAM worker is unavailable; inspect its environment/checkpoint")
        request_id = uuid.uuid4().hex
        message = {"protocol_version": PROTOCOL_VERSION, "request_id": request_id,
                   "action": action, **payload}
        try:
            assert self.process.stdin is not None
            self.process.stdin.write(json.dumps(message, allow_nan=False) + "\n")
            self.process.stdin.flush()
            deadline = time.monotonic() + self.timeout
            while True:
                if self.cancelled():
                    raise KeyboardInterrupt("SAM request cancelled")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SamWorkerError("SAM request timed out; reduce batch or check SAM environment")
                try:
                    line = self.lines.get(timeout=min(.1, remaining))
                    break
                except queue.Empty:
                    continue
            if line is None:
                raise SamWorkerError("SAM worker exited before replying")
            response = json.loads(line)
            if (not isinstance(response, dict) or response.get("protocol_version") != PROTOCOL_VERSION
                    or response.get("request_id") != request_id):
                raise SamWorkerError("SAM protocol version/request ID mismatch")
            if response.get("status") != "ok":
                message = str(response.get("error", "SAM failed"))
                if "out of memory" in message.lower():
                    message += "; reduce sam_batch_size and region_batch_size before retrying"
                raise SamWorkerError(message)
            return response
        except (OSError, ValueError) as error:
            self.close()
            raise SamWorkerError(f"Invalid SAM worker reply: {error}") from error
        except BaseException:
            self.close()
            raise

    def segment(self, images, directory):
        sources = [{key: value for key, value in asdict(source).items() if key not in ("view", "rules")}
                   for source in self.config.sources]
        response = self.request("segment", images=images, sources=sources,
                                directory=str(Path(directory).resolve()))
        self.last_metrics = response.get("metrics", {})
        records = response.get("records")
        if not isinstance(records, list):
            raise SamWorkerError("SAM reply is missing records")
        expected = {(image["image_id"], source.source_id) for image in images for source in self.config.sources}
        seen = set()
        dimensions = {image["image_id"]: tuple(image["size"]) for image in images}
        source_map = {source.source_id: source for source in self.config.sources}
        for record in records:
            key = (record["image_id"], record["source_id"])
            if key not in expected or key in seen or record.get("status") != "ok":
                raise SamWorkerError("SAM reply has unexpected, duplicate or failed image/source")
            seen.add(key)
            regions = record["regions"]
            if len(regions) > source_map[key[1]].max_regions:
                raise SamWorkerError("SAM exceeded region limit")
            for index, geometry in enumerate(regions):
                if (geometry["image_id"], geometry["source_id"], geometry["region_id"]) != (*key, index):
                    raise SamWorkerError("SAM region identity mismatch")
                if geometry.get("prompt") != source_map[key[1]].prompt:
                    raise SamWorkerError("SAM region prompt mismatch")
                finite_number(geometry.get("segmentation_score"), "segmentation_score", maximum=1)
                mask = read_mask(resolve_asset(directory, geometry["mask_path"]), dimensions[key[0]])
                if (mask_bbox(mask) != geometry["bbox"] or mask_digest(mask) != geometry["mask_sha256"]
                        or list(dimensions[key[0]]) != geometry["image_size"]):
                    raise SamWorkerError("SAM mask geometry/checksum mismatch")
        if seen != expected:
            raise SamWorkerError("SAM reply is incomplete")
        return records

    def close(self):
        process = self.process
        if process is None:
            return
        self.process = None
        self.closing.set()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for name in ("reader", "stderr_reader"):
            thread = getattr(self, name, None)
            if thread is not None:
                thread.join(timeout=1)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream:
                stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
