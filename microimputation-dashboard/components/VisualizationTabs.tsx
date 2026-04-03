'use client';

interface Tab {
  id: string;
  label: string;
  count?: number;
  description?: string;
}

interface VisualizationTabsProps {
  tabs: Tab[];
  activeTab: string;
  onTabChange: (tabId: string) => void;
}

export default function VisualizationTabs({
  tabs,
  activeTab,
  onTabChange,
}: VisualizationTabsProps) {
  return (
    <div className="border-b border-gray-200 mb-6">
      <nav className="-mb-px flex flex-wrap gap-x-4 gap-y-2 sm:gap-x-8" aria-label="Tabs">
        {tabs.map((tab) => (
          <div key={tab.id} className="relative group">
            <button
              onClick={() => onTabChange(tab.id)}
              className={`
                whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm
                ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }
              `}
            >
              {tab.label}
              {tab.count !== undefined && (
                <span
                  className={`
                    ml-2 py-0.5 px-2 rounded-full text-xs
                    ${
                      activeTab === tab.id
                        ? 'bg-blue-100 text-blue-600'
                        : 'bg-gray-100 text-gray-600'
                    }
                  `}
                >
                  {tab.count}
                </span>
              )}
            </button>
            {tab.description && (
              <div className="absolute left-1/2 -translate-x-1/2 bottom-full mb-2 px-3 py-2 bg-gray-900 text-white text-xs rounded-md whitespace-normal w-56 text-center opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-opacity z-50 pointer-events-none">
                {tab.description}
                <div className="absolute left-1/2 -translate-x-1/2 top-full w-0 h-0 border-l-4 border-r-4 border-t-4 border-l-transparent border-r-transparent border-t-gray-900" />
              </div>
            )}
          </div>
        ))}
      </nav>
    </div>
  );
}
