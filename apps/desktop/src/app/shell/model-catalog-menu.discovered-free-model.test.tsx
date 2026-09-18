import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { DropdownMenu, DropdownMenuContent } from '@/components/ui/dropdown-menu'
import { $localModelsEnabled } from '@/store/local-models-flag'
import { $localRuntimeJobs } from '@/store/local-runtime-jobs'
import { setModelVisibilityOpen, $visibleModels } from '@/store/model-visibility'

import { ModelCatalogMenu, type ModelMenuController } from './model-catalog-menu'

// Radix calls these on open; jsdom doesn't implement them.
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

// Transport only: the payload below is the REAL ModelOptionsResult shape the
// backend's build_model_options_payload produces after catalog discovery
// (curated list + a newly discovered free/tools model absent from curated).
// Nothing pre-computes the rendered list — the production data flow
// requestModelOptions -> useQuery -> groupModels -> render must surface it.
const getGlobalModelOptions = vi.fn()

vi.mock('@/hermes', () => ({
  getGlobalModelOptions: (...args: unknown[]) => getGlobalModelOptions(...args),
  getLocalModelsJobs: vi.fn(async () => {
    const { $localRuntimeJobs } = await import('@/store/local-runtime-jobs')
    return { jobs: [...$localRuntimeJobs.get()] }
  }),
  getLocalModelsStatus: vi.fn().mockResolvedValue({ loading: {} }),
  setApiRequestProfile: vi.fn()
}))

beforeEach(() => {
  $visibleModels.set(null)
  $localRuntimeJobs.set([])
  $localModelsEnabled.set(true)
  setModelVisibilityOpen(false)
  getGlobalModelOptions.mockResolvedValue({
    providers: [
      {
        slug: 'openrouter',
        name: 'OpenRouter',
        is_current: true,
        authenticated: true,
        // Curated entries first, then the DISCOVERED free/tools model the live
        // catalog surfaced and the backend appended (fetch_openrouter_models
        // appends unknown free tools models after curated, live order).
        models: [
          'fixture-lab/curated-a',
          'fixture-lab/curated-b',
          'fixture-lab/dynamic-free'
        ],
        total_models: 3,
        pricing: {
          'fixture-lab/curated-a': { input: 'free', output: 'free', free: true },
          'fixture-lab/curated-b': { input: 'free', output: 'free', free: true },
          'fixture-lab/dynamic-free': { input: 'free', output: 'free', free: true }
        },
        capabilities: {
          'fixture-lab/curated-a': { fast: false, reasoning: true },
          'fixture-lab/curated-b': { fast: true, reasoning: false },
          'fixture-lab/dynamic-free': { fast: true, reasoning: false }
        },
        source: 'built-in'
      }
    ]
  })
})

afterEach(() => {
  cleanup()
  $localRuntimeJobs.set([])
  vi.clearAllMocks()
})

function renderMenu() {
  const select = vi.fn()

  const controller: ModelMenuController = {
    applyPreset: vi.fn(),
    current: { effort: '', fast: false, model: 'fixture-lab/curated-a', provider: 'openrouter' },
    presetFor: () => ({}),
    select,
    setOptions: vi.fn()
  }

  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  render(
    <QueryClientProvider client={client}>
      <DropdownMenu open>
        <DropdownMenuContent>
          <ModelCatalogMenu controller={controller} />
        </DropdownMenuContent>
      </DropdownMenu>
    </QueryClientProvider>
  )

  return select
}

// FREE_MODEL_DESKTOP_PICKER_RENDER: a free, tool-compatible model newly
// discovered by the live catalog (absent from the curated static list) must
// reach the ACTUAL picker render through the production data flow.
describe('discovered free model reaches the desktop picker', () => {
  it('renders the discovered model row from a real model.options payload', async () => {
    renderMenu()

    // The discovered id is listed in the provider group (collapsed view keeps
    // the curated default set; search must surface the dynamic model).
    // displayModelName() renders ids title-cased ("Dynamic Free"), and the
    // search folds separators so the id-style query matches that label.
    const input = screen.getByRole('textbox', { name: 'Search models' })
    fireEvent.change(input, { target: { value: 'dynamic-free' } })

    await vi.waitFor(() => {
      expect(screen.getByText(/dynamic free/i)).toBeDefined()
    })
  })

  it('lists the discovered model among the provider rows (no search)', async () => {
    renderMenu()

    // Default visible set per provider (DEFAULT_VISIBLE_PER_PROVIDER) keeps the
    // first curated rows; with no shortlist every family renders, so the
    // discovered id appears alongside curated ones (ids render title-cased).
    await vi.waitFor(() => {
      expect(screen.getByText(/curated a/i)).toBeDefined()
      expect(screen.getByText(/curated b/i)).toBeDefined()
      expect(screen.getByText(/dynamic free/i)).toBeDefined()
    })
  })
})
