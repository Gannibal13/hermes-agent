import { describe, expect, it } from 'vitest'

import type { GatewayEventPayload } from '@/lib/chat-messages'

import {
  completionErrorText,
  delegateTaskPayloads,
  hasSessionInfoStatePatch,
  sessionInfoStatePatch,
  sessionRouteStatePatch,
  toTodoPayload
} from './utils'

const payload = (over: Record<string, unknown>): GatewayEventPayload => over as GatewayEventPayload

describe('sessionRouteStatePatch', () => {
  it('tracks simple → strong → simple without sticky fallback state', () => {
    const state = { model: 'simple', provider: 'primary', fallback: false, fallbackReason: '' }
    const strong = sessionRouteStatePatch(payload({ model: 'strong', provider: 'backup', fallback: true, fallback_reason: '429' }))
    const simple = sessionRouteStatePatch(payload({ model: state.model, provider: state.provider, fallback: false }))

    expect(strong).toEqual({ model: 'strong', provider: 'backup', fallback: true, fallbackReason: '429' })
    expect({ ...state, ...strong, ...simple }).toEqual(state)
  })

  it('preserves a manual provider/model pair and clears fallback reason', () => {
    expect(sessionRouteStatePatch(payload({ model: 'manual-model', provider: 'manual', fallback: false }))).toEqual({
      model: 'manual-model', provider: 'manual', fallback: false, fallbackReason: ''
    })
  })

  it('ignores a partial route instead of blanking the other half of the pair', () => {
    expect(sessionRouteStatePatch(payload({ model: 'model-without-provider', fallback: true }))).toBeNull()
    expect(sessionRouteStatePatch(payload({ provider: 'provider-without-model', fallback: true }))).toBeNull()
  })

  it('keeps compatibility with an older backend that sends no route fields', () => {
    expect(sessionRouteStatePatch(payload({ text: 'legacy reply', status: 'complete' }))).toBeNull()
  })
})

describe('completionErrorText', () => {
  it('flags provider/HTTP/retry failures, ignores normal text', () => {
    expect(completionErrorText('API call failed after 3 retries: boom')).toMatch(/^API call failed/)
    expect(completionErrorText('HTTP 500 upstream')).toMatch(/^HTTP 500/)
    expect(completionErrorText('Gateway error: nope')).toMatch(/^Gateway error/)
    expect(completionErrorText('here is your answer')).toBeNull()
    expect(completionErrorText('   ')).toBeNull()
  })
})

describe('toTodoPayload', () => {
  it('routes named todo and anonymous todos-bearing events to the todo stream', () => {
    expect(toTodoPayload(payload({ name: 'todo' }))?.tool_id).toBe('todo-live')
    expect(toTodoPayload(payload({ todos: [] }))?.name).toBe('todo_list')
    expect(toTodoPayload(payload({ name: 'todo_list' }))?.tool_id).toBe('todo-live')
    expect(toTodoPayload(payload({ name: 'web_search' }))).toBeUndefined()
    expect(toTodoPayload(undefined)).toBeUndefined()
  })
})

describe('sessionInfoStatePatch / hasSessionInfoStatePatch', () => {
  it('extracts only present runtime fields', () => {
    const patch = sessionInfoStatePatch(payload({ model: 'gpt', fast: true, branch: 'main' }))
    expect(patch).toMatchObject({ model: 'gpt', fast: true, branch: 'main' })
    expect(hasSessionInfoStatePatch(patch)).toBe(true)
    expect(hasSessionInfoStatePatch(sessionInfoStatePatch(payload({})))).toBe(false)
  })
})

describe('delegateTaskPayloads', () => {
  it('returns [] for non-delegate events', () => {
    expect(delegateTaskPayloads(payload({ name: 'web_search' }), 'running')).toEqual([])
  })

  it('maps a running tool.start to a subagent.start spec', () => {
    const [spec] = delegateTaskPayloads(
      payload({ name: 'delegate_task', tool_id: 't1', args: { goal: 'do it' } }),
      'running',
      'tool.start'
    )

    expect(spec).toMatchObject({ event_type: 'subagent.start', goal: 'do it', status: 'running' })
  })

  it('maps completion (with error) to a failed subagent.complete', () => {
    const [spec] = delegateTaskPayloads(
      payload({ name: 'delegate_task', error: 'boom', result: { summary: 'failed run' } }),
      'complete'
    )

    expect(spec).toMatchObject({ event_type: 'subagent.complete', status: 'failed' })
  })

  it.each(['timeout', 'error', 'failed', 'failure', 'TIMEOUT'])(
    'maps completion with result.status=%s to a failed subagent.complete',
    resultStatus => {
      const [spec] = delegateTaskPayloads(
        payload({ name: 'delegate_task', result: { status: resultStatus, summary: 'timed out' } }),
        'complete'
      )

      expect(spec).toMatchObject({ event_type: 'subagent.complete', status: 'failed' })
    }
  )

  it('maps a successful completion to completed', () => {
    const [spec] = delegateTaskPayloads(
      payload({ name: 'delegate_task', result: { status: 'success', summary: 'done' } }),
      'complete'
    )

    expect(spec).toMatchObject({ event_type: 'subagent.complete', status: 'completed' })
  })
})
