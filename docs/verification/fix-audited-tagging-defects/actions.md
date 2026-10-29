# Журнал реализации

Рабочая копия перенесена в Downloads; локальный launcher Python восстановлен отдельным окружением apply-runtime.
Исторические результаты ревизии сохранены без перезаписи.

Контракты: TXT/read-failure и before/after → группа 2; пустая ветвь, порядок, character, нормализация, caption и global add/remove → группа 3; пропуски и basename → группа 4; NameError → группа 5.

- 1.1: baseline.json: 30 tests, 18 passed, 11 failures, 1 error; before-hashes.json saved.

- 1.2: All 12 failing contracts mapped to task groups; original assertions retained.

- 2.1: Read failure and fsync/replace injection preserve original bytes; temporary files cleaned.

- 2.2: TagManager uses the same writer; mixed-success save counts only successful file.

- 2.3: Mixed delimiters, canonical deduplication, before/after and repeat append pass.

- 2.4: SafeStorage 4/4 pass; original audit improves 18 -> 20 passed; behavior documented.

- 3.1: Canonical formatting and punctuation tested; strings reflect score dictionaries.

- 3.2: Order, empty face, maximum score and input immutability contracts pass.

- 3.3: Region helmet removal and spaced canonical names reflected in caption and dictionaries.

- 3.4: Global add/remove and before/after share formatting; original audit 27/30 pass.

- 3.5: Manual known character, unknown general and original scores tested; remove wins.

- 3.6: Filter behavior documented; baseline notebook entry and all tag contracts pass.

- 4.1: Recursive labels, roots, duplicate basenames, repeated IDs and outside-root tests pass.

- 4.2: All labelled images participate in metrics and both tables; missing prediction lowers Recall.

- 4.3: Optional diagnostics contain coverage and counts; default metric keys unchanged.

- 4.4: Empty label sets, empty datasets and unknown prediction IDs tested.

- 4.5: Historical harmonic-of-means F1 tested and explicitly labelled; notebook return preserved.

- 5.1: Original NameError contract and actual PNG export without Plotly static renderer pass.

- 5.2: Empty detection and inference exception release files; detached crops remain readable.

- 5.3: Resource regressions pass with ResourceWarning promoted; behavior documented.

- 6.1: Final regressions.json: 49/49 passed, zero errors/failures/skips; unchanged original 30 and updated audit/README.md; original historical audit preserved.

- 6.2: Mutator attempted twice: Windows WinError 1314, not counted as kills. Documented isolated-copy fallback: 2/2 selected semantic mutations caught by assertions; import paths captured; no equivalent/surviving selected cases.

- Integration acceptance found DataFrame.map unsupported by pinned pandas 2.0.3. New displayed-table regression reproduced AttributeError, then two numeric-table formatters switched to compatible Series.map; public API unchanged.

- 6.3: Pylint error/fatal=0; warnings 46 vs 71, no new warning signatures. Crapper uses passing-test LCOV; CC reductions 32->19, 20->10, 18->6. Existing visual-analytics duplicate retained per design; CRAP/coverage explicitly reported.

- 6.4: WD ViT v3 (448x448), CPU, current person/face YOLO: 5/5 copied images including user reference. Separate before/after: eyelashes restored for empty-face case; other merged tag sets unchanged. Original images/TXT and reference hashes preserved.

- 6.5: Real CPU BatchTagging signature/two lists, JSON round-trip, idempotent TXT append, genuine PNG/JPEG, Plotly, notebook 5-tuple verified. Final 50/50 tests, integrity hashes, environment and limitations saved in acceptance README; all actions logged.
