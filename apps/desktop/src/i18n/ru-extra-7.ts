import type { TranslationOverrides } from './define-locale'

// Русификация: Навыки, загрузка, меню файлов, заголовок, правая панель.
export const ruExtra7: TranslationOverrides = {
  fileMenu: {
    download: 'Скачать',
    downloadSaved: 'Сохранено',
    downloadFailed: 'Не удалось скачать'
  },
  boot: {
    steps: {
      retryingRemoteBackend: 'Переподключение к удалённому бэкенду Hermes…'
    },
    failure: {
      remoteSignInHint: signInLabel => `Выход из сохранённой браузерной сессии удалённого доступа, затем открывается «${signInLabel}». Чтобы переключиться на встроенный бэкенд, используйте локальный шлюз.`,
      cloudDownTitle: 'Облачный агент Nous недоступен',
      cloudDownDescription: 'Управляемый Nous облачный агент, к которому подключён этот шлюз, отдаёт ошибку сервера. Перезапустить его отсюда нельзя — проверьте его статус, переключитесь на локальный шлюз или обратитесь в поддержку.',
      cloudDownHint: 'Кнопки ниже открывают портал Nous (статус и управление экземпляром) и наш Discord для поддержки.',
      cloudDownCheckPortal: 'Статус в Портале',
      cloudDownDiscord: 'Помощь в Discord'
    }
  },
  titlebar: {
    unreadSessions: count => (count === 1 ? '1 непрочитанная сессия' : `${count} непрочитанных сессий`)
  },
  skills: {
    visionModelHint: 'Зрение использует конфигурацию вспомогательных моделей — модель с поддержкой изображений выбирается там, а не здесь по провайдерам.',
    visionModelLink: 'Выбрать модель зрения в Настройки → Модели',
    toolsetsEnabled: (enabled, total) => `Включено наборов: ${enabled}/${total}`,
    configureToolset: label => `Настроить ${label}`,
    toggleToolset: (label, enabled) => `${enabled ? 'Включить' : 'Выключить'} набор ${label}`,
    skillsLoadFailed: 'Не удалось загрузить навыки',
    toolsetsRefreshFailed: 'Не удалось обновить наборы инструментов',
    skillEnabled: 'Навык включён',
    skillDisabled: 'Навык выключен',
    toolsetEnabled: 'Набор инструментов включён',
    toolsetDisabled: 'Набор инструментов выключен',
    appliesToNewSessions: name => `${name} применяется к новым сессиям.`,
    failedToUpdate: name => `Не удалось обновить ${name}`,
    sortMostUsed: 'Самые используемые',
    sortAlpha: 'А–Я',
    sortMostUsedDesc: '↓ Самые используемые',
    sortLeastUsedAsc: '↑ Наименее используемые',
    bulkUpdated: count => `Обновлено для новых сессий: ${count}.`,
    bulkNoChange: 'Изменять нечего.',
    usageCount: count => `использован ${count}×`,
    provenance: {
      agent: 'Изучен',
      bundled: 'Встроенный',
      hub: 'Хаб'
    },
    emptyNoneFound: noun => `${noun} не найдено`,
    emptyNothingMatches: query => `Ничего не найдено по «${query}».`,
    emptyNoneAvailable: noun => `${noun} пока нет.`,
    changesApplyNewSessions: 'Изменения применяются к новым сессиям.',
    skillUpdated: 'Навык обновлён',
    skillArchivedTitle: 'Навык отправлен в архив',
    skillArchivedMessage: 'Можно восстановить через hermes curator restore.',
    hub: {
      searchPlaceholder: 'Поиск по хабу навыков',
      search: 'Искать',
      searching: 'Поиск...',
      connectingHubs: 'Подключение к хабам навыков...',
      connectedHubs: 'Подключённые хабы:',
      featured: 'Рекомендуемые навыки',
      landingHint: 'Ищите в хабе, чтобы просматривать устанавливаемые навыки из официального индекса, GitHub и сообщества.',
      noResults: 'В хабе нет подходящих навыков.',
      resultCount: (count, ms) => `Результатов: ${count}${ms !== null ? ` за ${ms} мс` : ''}`,
      timedOut: sources => `Истёк тайм-аут: ${sources}`,
      installed: 'Установлен',
      install: 'Установить',
      installing: 'Установка...',
      uninstall: 'Удалить',
      uninstalling: 'Удаление...',
      updateAll: 'Обновить установленные',
      updating: 'Обновление...',
      preview: 'Просмотр',
      scan: 'Сканировать',
      scanning: 'Сканирование...',
      close: 'Закрыть',
      files: 'Файлы',
      noReadme: 'У этого навыка нет SKILL.md для просмотра.',
      trust: {
        builtin: 'встроенный',
        trusted: 'доверенный',
        community: 'сообщество'
      },
      verdictSafe: 'Безопасно',
      verdictCaution: 'Осторожно',
      verdictDangerous: 'Опасно',
      policyAllow: 'Установка разрешена',
      policyAsk: 'Проверьте перед установкой',
      policyBlock: 'Установка заблокирована политикой',
      findings: count => `Находок: ${count}`,
      noFindings: 'Проблем безопасности не найдено.',
      installStarted: name => `Устанавливаю ${name}...`,
      uninstallStarted: name => `Удаляю ${name}...`,
      updateStarted: 'Обновляю установленные навыки...',
      actionFailed: 'Действие с навыком не удалось',
      actionLog: 'Журнал действий',
      alreadyInstalled: name => `«${name}» уже установлен`,
      pickerTitle: 'Хаб навыков',
      pickerBrowse: 'Открыть весь хаб',
      pickerHide: 'Скрыть обозреватель хаба',
      pickerHint: 'Нажмите «+ Добавить этому агенту» на любом навыке — он установится и появится в списке выше.',
      loadFailed: 'Не удалось загрузить хаб навыков',
      previewFailed: 'Не удалось открыть предпросмотр навыка',
      scanFailed: 'Сканирование безопасности не удалось',
      searchFailed: 'Поиск по хабу не удался'
    }
  },
  rightSidebar: {
    folderTip: cwd => cwd,
    couldNotPreview: path => `Не удалось показать ${path}`,
    unreadableTitle: 'Нечитаемо',
    unreadableBody: error => `Не удалось прочитать эту папку (${error}).`,
    treeErrorTitle: 'Ошибка дерева файлов',
    treeErrorBody: 'При отрисовке этой папки произошла ошибка.'
  }
}
