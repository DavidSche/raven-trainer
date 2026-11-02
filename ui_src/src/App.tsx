import React, { Component, Suspense, lazy, type ReactNode } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AlertTriangle, RefreshCw, Loader2 } from 'lucide-react';
import i18n from './i18n';
import { Sidebar } from './components/layout/Sidebar';
import { TopBar } from './components/layout/TopBar';

// ── Route-level lazy loading ──
// Each page is split into its own chunk so the initial bundle stays small.
// The Sidebar/TopBar layout shell loads eagerly; pages load on-demand.
const DashboardPage = lazy(() => import('./pages/DashboardPage'));
const DatasetsPage = lazy(() => import('./pages/DatasetsPage'));
const DatasetDetailPage = lazy(() => import('./pages/DatasetDetailPage'));
const DatasetBrowsePage = lazy(() => import('./pages/DatasetBrowsePage'));
const RecipeListPage = lazy(() => import('./pages/RecipeListPage'));
const RecipeDetailPage = lazy(() => import('./pages/RecipeDetailPage'));
const WorkspacesPage = lazy(() => import('./pages/WorkspacesPage'));
const IngestionProfilesPage = lazy(() => import('./pages/IngestionProfilesPage'));
const StorageSourcesPage = lazy(() => import('./pages/StorageSourcesPage'));
const StorageSourceDetailPage = lazy(() => import('./pages/StorageSourceDetailPage'));
const TrainingPage = lazy(() => import('./pages/TrainingPage'));
const TrainingDetailPage = lazy(() => import('./pages/TrainingDetailPage'));
const PipelinePage = lazy(() => import('./pages/PipelinePage'));
const JobsPage = lazy(() => import('./pages/JobsPage'));
const ModelsPage = lazy(() => import('./pages/ModelsPage'));
const TasksPage = lazy(() => import('./pages/TasksPage'));
const MisdataPage = lazy(() => import('./pages/MisdataPage'));
const ReportsPage = lazy(() => import('./pages/ReportsPage'));
const SettingsPage = lazy(() => import('./pages/SettingsPage'));
const NotFoundPage = lazy(() => import('./pages/NotFoundPage'));

/** Full-page loading fallback shown while a lazy route chunk is being fetched. */
function RouteLoadingFallback() {
  return (
    <div className="flex items-center justify-center h-full py-20">
      <Loader2 className="w-6 h-6 text-blue-500 animate-spin" />
    </div>
  );
}

const queryClient = new QueryClient();

const basename = (window as unknown as Record<string, unknown>).__RAVEN_CONSOLE_BASE
  ? '/trainer'
  : undefined;

// ── Global Error Boundary ──
interface EBState { hasError: boolean; error: Error | null; }
class ErrorBoundary extends Component<{ children: ReactNode }, EBState> {
  state: EBState = { hasError: false, error: null };
  static getDerivedStateFromError(error: Error): EBState { return { hasError: true, error }; }
  render() {
    if (!this.state.hasError) return this.props.children;
    return (
      <div className="flex items-center justify-center h-screen bg-gray-50">
        <div className="text-center p-8 max-w-md">
          <AlertTriangle className="w-12 h-12 text-amber-500 mx-auto mb-4" />
          <h2 className="text-lg font-semibold text-gray-800 mb-2">{i18n.t('errorBoundary.title')}</h2>
          <p className="text-sm text-gray-500 mb-4">{this.state.error?.message || i18n.t('errorBoundary.fallbackMessage')}</p>
          <button onClick={() => window.location.reload()}
            className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700">
            <RefreshCw className="w-4 h-4" /> {i18n.t('errorBoundary.reload')}
          </button>
        </div>
      </div>
    );
  }
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ErrorBoundary>
      <BrowserRouter basename={basename}>
        <div className="flex h-screen overflow-hidden">
          {!(window as unknown as Record<string, unknown>).__RAVEN_CONSOLE_EMBED && <Sidebar />}
          <div className="flex-1 flex flex-col min-w-0">
            {!(window as unknown as Record<string, unknown>).__RAVEN_CONSOLE_EMBED && <TopBar />}
            <main className="flex-1 overflow-y-auto bg-gray-50">
              <Suspense fallback={<RouteLoadingFallback />}>
                <Routes>
                  <Route path="/" element={<DashboardPage />} />
                  <Route path="/datasets" element={<DatasetsPage />} />
                  <Route path="/datasets/:name" element={<DatasetDetailPage />} />
                  <Route path="/datasets/:name/browse/:version" element={<DatasetBrowsePage />} />
                  <Route path="/recipes" element={<RecipeListPage />} />
                  <Route path="/recipes/:name" element={<RecipeDetailPage />} />
                  <Route path="/workspaces" element={<WorkspacesPage />} />
                  <Route path="/ingestion-profiles" element={<IngestionProfilesPage />} />
                  <Route path="/training" element={<TrainingPage />} />
                  <Route path="/storage" element={<StorageSourcesPage />} />
                  <Route path="/storage/:media_type/:source" element={<StorageSourceDetailPage />} />
                  <Route path="/training/:id" element={<TrainingDetailPage />} />
                  <Route path="/pipeline" element={<PipelinePage />} />
                  <Route path="/jobs" element={<JobsPage />} />
                  <Route path="/models" element={<ModelsPage />} />
                  <Route path="/tasks" element={<TasksPage />} />
                  <Route path="/misdata" element={<MisdataPage />} />
                  <Route path="/reports" element={<ReportsPage />} />
                  <Route path="/settings" element={<SettingsPage />} />
                  <Route path="*" element={<NotFoundPage />} />
                </Routes>
              </Suspense>
            </main>
          </div>
        </div>
      </BrowserRouter>
      </ErrorBoundary>
    </QueryClientProvider>
  );
}
