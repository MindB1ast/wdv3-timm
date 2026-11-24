# Tasks

## 1. Автоматические проверки

- [x] 1.1 Добавить `.github/workflows/ci.yml` с `push` и `pull_request` для `main`, независимыми задачами, доступом только на чтение, таймаутами и отменой устаревших прогонов. Проверить структуру и синтаксис workflow валидатором GitHub Actions.
- [x] 1.2 Настроить CPU-окружение Python 3.11 и запустить `audit.test_saved_region_filters`, `audit.test_region_filter_outcomes`, `audit.test_conditional_tags`, `audit.test_folder_postprocessing`, `audit.test_comfy_export`, `audit.test_cascade_tags`, `audit.test_tag_policies` и `audit.test_tag_regex`. Подтвердить запуск без ComfyUI, локальных моделей и доступа к сохранённым результатам вне Git.
- [x] 1.3 Закрепить доступную публичную версию OpenSpec CLI и выполнить `openspec validate --specs`; подтвердить корректный результат команды и передачу ненулевого кода завершения в Actions.
- [x] 1.4 Записать в `audit/README.md` эквивалентные локальные команды, назначение двух проверок и отличие от ручных запусков с моделями. Сверить команды с workflow.

## Workflow follow-up

- После публикации workflow проверить, что GitHub Actions выполнил обе задачи, и сообщить результат отдельно от локальных проверок.
