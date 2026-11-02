import React, { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { BookTemplate, Loader2 } from 'lucide-react';
import RecipeBuilderTab from '../components/dataset/RecipeBuilderTab';
import { LoadingState } from '../components/common/StateViews';

export default function RecipeDetailPage() {
  const { t } = useTranslation();
  const { name } = useParams<{ name: string }>();
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (name) setReady(true);
  }, [name]);

  if (!ready || !name) {
    return <div className="p-6"><LoadingState /></div>;
  }

  return (
    <div className="p-6 max-w-6xl mx-auto">
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 text-sm text-gray-500 mb-4">
        <Link to="/recipes" className="hover:text-purple-600 flex items-center gap-1">
          <BookTemplate className="w-4 h-4" /> {t('nav.recipes')}
        </Link>
        <span>/</span>
        <span className="text-gray-900 font-medium">{name}</span>
        <span className="text-[10px] text-purple-500 bg-purple-50 px-2 py-0.5 rounded ml-2">
          {t('recipesDetail.standaloneLabel')}
        </span>
      </div>

      {/* Three-layer concept banner */}
      <div className="text-xs bg-purple-50 border border-purple-200 rounded-lg px-4 py-2.5 text-purple-700 mb-4 leading-relaxed">
        {t('recipesDetail.standaloneDesc')} <code className="bg-purple-100 px-1 rounded">data/recipes/{name}/recipe.yaml</code>.
        {t('recipesDetail.multiSourceHint')}
      </div>

      <RecipeBuilderTab recipeName={name} />
    </div>
  );
}
