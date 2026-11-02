import { useTranslation } from 'react-i18next';
import { Link } from 'react-router-dom';
import { Home, Frown } from 'lucide-react';

export default function NotFoundPage() {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col items-center justify-center h-full py-20">
      <Frown className="w-12 h-12 text-gray-300 mb-4" />
      <h1 className="text-lg font-semibold text-gray-700 mb-2">404</h1>
      <p className="text-sm font-medium text-gray-500 mb-1">{t('common.notFound')}</p>
      <p className="text-xs text-gray-400 mb-6">{t('common.notFoundHint')}</p>
      <Link
        to="/"
        className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700 transition-colors"
      >
        <Home className="w-4 h-4" /> {t('nav.dashboard')}
      </Link>
    </div>
  );
}
