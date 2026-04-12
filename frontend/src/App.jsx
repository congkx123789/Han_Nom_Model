import React, { useEffect, useState } from 'react';
import { Database } from 'lucide-react';

// Common Components
import Sidebar from './components/common/Sidebar';
import Navbar from './components/common/Navbar';
import AuthModal from './components/common/AuthModal';

// Admin Views (Researcher)
import Workspace from './views/admin/Workspace';
import Archive from './views/admin/Archive';
import Admin from './views/admin/Admin';
import ChatAssistant from './views/admin/ChatAssistant';

// Client Views (Public)
import ClientPortal from './views/client/ClientPortal';
import Discovery from './views/client/Discovery';
import FloatingAIBubble from './components/common/FloatingAIBubble';

const App = () => {
  const [activeTab, setActiveTab] = useState('portal'); // 'portal' is the new default landing
  const [isResearcherMode, setIsResearcherMode] = useState(false);
  const [darkMode, setDarkMode] = useState(true);
  const [user, setUser] = useState(null);
  const [showAuth, setShowAuth] = useState(false);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode);
  }, [darkMode]);

  // If user switches to researcher mode, set a researcher-specific tab
  const toggleMode = () => {
    const nextMode = !isResearcherMode;
    setIsResearcherMode(nextMode);
    if (nextMode) {
      setActiveTab('chatbot'); // Default for researcher
    } else {
      setActiveTab('portal'); // Default for client
    }
  };

  return (
    <div className={`app-shell ${!isResearcherMode ? 'client-layout' : ''}`}>
      {/* Sidebar only appears in Researcher/Admin mode */}
      {isResearcherMode && <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} user={user} setShowAuth={setShowAuth} />}

      <div className="app-main">
        <Navbar
          darkMode={darkMode}
          setDarkMode={setDarkMode}
          user={user}
          setShowAuth={setShowAuth}
          isResearcherMode={isResearcherMode}
          onToggleMode={toggleMode}
        />

        <main className="main-content">
          {/* Client Routes */}
          {!isResearcherMode && activeTab === 'portal' && <ClientPortal onSearch={(q) => { setActiveTab('discovery'); }} />}
          {!isResearcherMode && activeTab === 'discovery' && <Discovery />}

          {/* Admin Routes */}
          {isResearcherMode && activeTab === 'workspace' && <Workspace />}
          {isResearcherMode && activeTab === 'archive' && <Archive />}
          {isResearcherMode && activeTab === 'admin' && <Admin />}
          {isResearcherMode && activeTab === 'dictionary' && <Discovery />}
          {isResearcherMode && activeTab === 'chatbot' && <ChatAssistant />}

          {/* Empty State Fallback */}
          {!['portal', 'discovery', 'workspace', 'archive', 'admin', 'dictionary', 'chatbot'].includes(activeTab) && (
            <div className="empty-state">
              <Database size={58} />
              <p>Màn hình "{activeTab}" đang được hoàn thiện.</p>
            </div>
          )}
        </main>
      </div>

      <AuthModal
        isOpen={showAuth}
        onClose={() => setShowAuth(false)}
        onLogin={(username) => {
          setUser(username);
          setShowAuth(false);
        }}
      />

      {/* Floating AI Bubble only for Client Mode */}
      {!isResearcherMode && <FloatingAIBubble />}
    </div>
  );
};

export default App;
