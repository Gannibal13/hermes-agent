# ACTIVE CONTRACT — Smart Router Reference Ready
MinEvidenceItems: 20

Каждый обязательный пункт имеет статус и доказательство уровня assert.
Финал разрешён только прогоном `gate_smart_router.py` с кодом 0.
Зелёный тест без привязки к требуемому поведению доказательством не является.

Формат блоков машиночитаем: гейт парсит `## [ID]`, строки `- tests/...::...`
в секции `Evidence:` и поле `Asserts:`. Не меняй формат без обновления гейта.

## [P1] Маршрут выбирается политикой, а не наследованием от родителя
Status: CLOSED
Evidence:
- tests/tools/test_delegate_smart_routing.py::TestDelegateSmartRouting::test_mechanical_child_does_not_inherit_sol
Asserts: mechanical-ребёнок Sol-родителя получает mini-flash-cheap/openrouter через реальный _build_child_agent, а не модель родителя.

## [P2] Дешёвая альтернатива предпочтительнее Sol по умолчанию
Status: CLOSED
Evidence:
- tests/agent/test_smart_router.py::TestCheapFirst::test_mechanical_ignores_sol
Asserts: select_route([sol, cheap, standard], mechanical) возвращает mini-flash-cheap.

## [P3] Null означает авто-выбор через роутер, а не наследование
Status: CLOSED
Evidence:
- tests/agent/test_smart_router.py::TestAutoRoute::test_null_means_auto_not_inherit
Asserts: model=None при наличии дешёвого кандидата даёт mini-flash-cheap, а не модель родителя.

## [P4] Роутер учитывает сложность задачи
Status: CLOSED
Evidence:
- tests/agent/test_smart_router.py::TestComplexity::test_mechanical_by_default
- tests/agent/test_smart_router.py::TestComplexity::test_standard
- tests/agent/test_smart_router.py::TestComplexity::test_complex
Asserts: typo→mechanical, OAuth→standard, deadlock→complex.

## [P5] Роутер учитывает размер текущего контекста
Status: CLOSED
Evidence:
- tests/agent/test_smart_router.py::TestContextAndBudget::test_large_context_filtered_before_routing
- tests/agent/test_smart_router.py::TestContextAndBudget::test_never_131k_for_ordinary_task
Asserts: ~96k токенов режутся до cap до выбора; обычный таск держит kept_tokens ниже абсолютного потолка.

## [P6] Роутер учитывает квоту
Status: CLOSED
Evidence:
- tests/agent/test_smart_router.py::TestCheapFirst::test_quota_exhausted_skipped
Asserts: маршрут с quota_remaining=0 пропускается даже при нулевой цене.

## [P7] Роутер учитывает стоимость
Status: CLOSED
Evidence:
- tests/agent/test_smart_router.py::TestCheapFirst::test_mechanical_ignores_sol
- tests/agent/test_smart_router.py::TestCheapFirst::test_standard_avoids_strong
Asserts: mechanical/standard выбирают cheapest suitable, а не первый попавшийся.

## [P8] Роутер учитывает реальную пригодность модели
Status: CLOSED
Evidence:
- tests/agent/test_smart_router.py::TestCheapFirst::test_small_window_skipped
- tests/agent/test_smart_router.py::TestCheapFirst::test_standard_avoids_strong
Asserts: окно < need_tokens и strong-тир для обычной работы отбрасываются.

## [P9] Сначала дешёвая/local для механической работы
Status: CLOSED
Evidence:
- tests/agent/test_smart_router.py::TestCheapFirst::test_mechanical_ignores_sol
- tests/agent/test_smart_router.py::TestCheapFirst::test_local_preferred
Asserts: local-маршрут побеждает remote-cheap при равной пригодности.

## [P10] Эскалация на сильную только для сложного рассуждения
Status: CLOSED
Evidence:
- tests/agent/test_smart_router.py::TestCheapFirst::test_complex_escalates_to_strong
- tests/agent/test_smart_router.py::TestCheapFirst::test_complex_falls_back_when_no_strong
Asserts: complex при живых cheap выбирает strong; без strong — лучший из оставшихся, а не ошибка.

## [P11] После эскалации — обратно на подходящую
Status: CLOSED
Evidence:
- tests/tools/test_delegate_smart_routing.py::TestDelegateSmartRouting::test_escalate_then_back_down
Asserts: complex→Sol, следующий mechanical→cheap: эскалация не залипает (stateless).

## [P12] Sol не дефолт без причины
Status: CLOSED
Evidence:
- tests/agent/test_smart_router.py::TestCheapFirst::test_mechanical_ignores_sol
- tests/tools/test_delegate_smart_routing.py::TestDelegateSmartRouting::test_no_alternatives_keeps_parent_with_reason
Asserts: в модели нет 'sol' при наличии альтернатив; Sol сохраняется лишь когда альтернатив нет, с reason.

## [P13] Многошаговый фейловер A→B→C→D до исчерпания
Status: CLOSED
Evidence:
- tests/agent/test_smart_router.py::TestFailover::test_walk_a_b_c_d_to_success
- tests/agent/test_smart_router.py::TestFailoverMatrix::test_failover_matrix
- tests/tools/test_delegate_smart_routing.py::TestDelegateSmartRouting::test_smart_routed_child_keeps_parent_fallback_chain
- tests/tools/test_delegate.py::TestFallbackModelInheritance::test_child_inherits_fallback_chain
Asserts: 4 попытки по порядку, победа на D, лог из 4 записей; авто-маршрут не гасит цепочку (иначе фейловер схлопывается).

## [P14] 402/429/оффлайн/auth не останавливают при живых маршрутах
Status: CLOSED
Evidence:
- tests/agent/test_smart_router.py::TestFailover::test_retryable_errors
- tests/agent/test_smart_router.py::TestFailoverMatrix::test_failover_matrix
Asserts: 402/429/offline/401/auth-retryable; матрица проходит billing→rate-limit→auth→success.

## [P15] Неповторяемая ошибка стопает сразу, не жгёт цепочку
Status: CLOSED
Evidence:
- tests/agent/test_smart_router.py::TestFailover::test_non_retryable_stops_fast
Asserts: 400 останавливает на первой попытке, B не вызывается.

## [P16] Полная недоступность — громкая ошибка, не тихий успех
Status: CLOSED
Evidence:
- tests/agent/test_smart_router.py::TestFailover::test_full_unavailable_raises
- tests/agent/test_smart_router.py::TestFailover::test_empty_chain_raises
Asserts: все маршруты мертвы / цепочка пуста → исключение.

## [P17] Большой контекст фильтруется ДО выбора модели
Status: CLOSED
Evidence:
- tests/agent/test_smart_router.py::TestContextAndBudget::test_large_context_filtered_before_routing
- tests/agent/test_smart_router.py::TestContextAndBudget::test_small_context_passes_through
Asserts: большой режется с флагом truncated, маленький идёт как есть без потерь.

## [P18] Обычный таск не запрашивает ~131k токенов
Status: CLOSED
Evidence:
- tests/agent/test_smart_router.py::TestContextAndBudget::test_never_131k_for_ordinary_task
Asserts: kept_tokens < ABSOLUTE_CONTEXT_TOKEN_CEILING и auto_compress включён.

## [P19] Адаптивный бюджет вывода; авто-режим без блоков
Status: CLOSED
Evidence:
- tests/agent/test_smart_router.py::TestContextAndBudget::test_adaptive_budgets
- tests/agent/test_smart_router.py::TestContextAndBudget::test_auto_never_blocks
Asserts: 2000 < 4000 < 8000, cap 8000; blocking_confirm всегда False в авто.

## [P20] Ручная модель/пин всегда побеждает роутер
Status: CLOSED
Evidence:
- tests/agent/test_smart_router.py::TestAutoRoute::test_explicit_pin_wins
- tests/agent/test_smart_router.py::TestAutoRoute::test_manual_model_honored
- tests/tools/test_delegate_smart_routing.py::TestDelegateSmartRouting::test_explicit_task_model_honored
- tests/tools/test_delegate_smart_routing.py::TestDelegateSmartRouting::test_delegation_model_pin_honored
- tests/tools/test_delegate_smart_routing.py::TestDelegateSmartRouting::test_router_disabled_inherits_parent
Asserts: явная модель и delegation.model проходят untouched, smart-блок пропускается; smart_router:false возвращает наследование.

## [P21] E2E-набор делегации зелёный целиком
Status: CLOSED
Evidence:
- tests/tools/test_delegate_smart_routing.py::TestDelegateSmartRouting::test_mechanical_child_does_not_inherit_sol
- tests/tools/test_delegate_smart_routing.py::TestDelegateSmartRouting::test_explicit_task_model_honored
- tests/tools/test_delegate_smart_routing.py::TestDelegateSmartRouting::test_delegation_model_pin_honored
- tests/tools/test_delegate_smart_routing.py::TestDelegateSmartRouting::test_router_disabled_inherits_parent
- tests/tools/test_delegate_smart_routing.py::TestDelegateSmartRouting::test_no_alternatives_keeps_parent_with_reason
- tests/tools/test_delegate_smart_routing.py::TestDelegateSmartRouting::test_smart_routed_child_keeps_parent_fallback_chain
- tests/tools/test_delegate_smart_routing.py::TestDelegateSmartRouting::test_escalate_then_back_down
Asserts: все 7 e2e через реальный _build_child_agent, включая прикреплённый route log и бюджет.

## [P22] Лог маршрутов реальный: что tried, исход, ошибка, время
Status: CLOSED
Evidence:
- tests/agent/test_smart_router.py::TestAutoRoute::test_route_log_records
- tests/tools/test_delegate_smart_routing.py::TestDelegateSmartRouting::test_mechanical_child_does_not_inherit_sol
Asserts: записи success/failed с текстом ошибки и ts; ребёнок несёт непустой _smart_route_log.

## [P23] Финальный доказательный отчёт — это вывод гейта
Status: CLOSED
Evidence:
- GATE:SELF
Asserts: отчёт существует только как построчный PASS/FAIL прогона гейта; отдельный prose-отчёт доказательством не является.

## [M1] У каждого пункта есть статус и доказательство
Status: CLOSED
Evidence:
- GATE:STRUCT
Asserts: гейт парсит контракт и требует у каждого блока Status + ≥1 Evidence-nodeid + Asserts.

## [M2] Преждевременный финал реально блокируется
Status: CLOSED
Evidence:
- GATE:SELFTEST
Asserts: каждый прогон гейта выполняет негативный контроль (force-open) и падает, если блок не срабатывает.

## [M3] Evidence обязано быть исполнено в этом прогоне
Status: CLOSED
Evidence:
- GATE:STRUCT
Asserts: гейт запускает ровно объединённое множество nodeid из контракта и сверяет покрытие.

## [M4] Требования не схлопываются в один тест
Status: CLOSED
Evidence:
- GATE:STRUCT
Asserts: P-пунктов ≥20, у каждого ≥1 nodeid, наборы nodeid попарно не идентичны.

## [M5] Старые тесты не дают PASS автоматом — только свежий прогон без кэша
Status: CLOSED
Evidence:
- GATE:STRUCT
Asserts: гейт всегда выполняет pytest сам с -p no:cacheprovider и принимает только PASSED этого прогона.

## Amendments (уточнения мёржатся в контракт, а не начинают заново)
- A1: complex залипал на дешёвых (cost-first) → select_route: strong tier first; тесты test_complex_escalates_to_strong, test_complex_falls_back_when_no_strong.
- A2: запрошена матрица фейловера 402/429/auth→D → TestFailoverMatrix::test_failover_matrix.
- A3: запрошен контрактный гейт с запретом финала → этот файл + gate_smart_router.py.
- A4: запрошена глобальная Execution Contract System → contracts/gate.py (универсальный гейт), contracts/README.md (протокол), gate_smart_router.py стал тонкой обёрткой; контракт мигрирован без сужения (все 28 пунктов сохранены).
