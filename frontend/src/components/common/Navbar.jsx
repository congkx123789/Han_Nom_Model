import React from 'react';
import { Search, Bell, BookOpenText, Sun, Moon, ChevronDown, LayoutDashboard, Globe } from 'lucide-react';

const Navbar = ({ darkMode, setDarkMode, user, setShowAuth, isResearcherMode, onToggleMode }) => {
  return (
    <header className="top-header">
      <div className="header-left">
        <BookOpenText size={18} className="brand-icon" />
        <span className="brand-name">HÁN NÔM HERITAGE</span>
      </div>

      <div className="header-center">
        {!isResearcherMode ? (
          <div className="client-nav-links">
            <button className="nav-link active">Khám phá</button>
            <button className="nav-link">Thư viện số</button>
            <button className="nav-link">Về dự án</button>
          </div>
        ) : (
          <div className="search-bar-container">
            <Search size={16} className="search-icon" />
            <input
              type="text"
              placeholder="Tìm kiếm di sản, thư tịch, âm Hán..."
              className="global-search"
            />
          </div>
        )}
      </div>

      <div className="header-right">
        {/* MODE TOGGLE SWITCH */}
        <button
          className={`mode-toggle-btn ${isResearcherMode ? 'scholar' : 'public'}`}
          onClick={onToggleMode}
          title={isResearcherMode ? "Chuyển sang Cổng thông tin công cộng" : "Chuyển sang Không gian Nghiên cứu"}
        >
          {isResearcherMode ? <Globe size={16} /> : <LayoutDashboard size={16} />}
          <span>{isResearcherMode ? 'PUBLIC PORTAL' : 'RESEARCHER MODE'}</span>
        </button>

        <button className="header-icon-btn" onClick={() => setDarkMode(!darkMode)} title="Thay đổi giao diện">
          {darkMode ? <Sun size={18} /> : <Moon size={18} />}
        </button>

        {isResearcherMode && (
          <button className="header-icon-btn notify-btn" title="Thông báo">
            <Bell size={18} />
            <span className="notify-badge">3</span>
          </button>
        )}

        {user ? (
          <button className="profile-menu-btn">
            <div className="user-avatar-sm">{user.substring(0, 1).toUpperCase()}</div>
            <span>{user}</span>
            <ChevronDown size={14} />
          </button>
        ) : (
          <button className="login-trigger-btn" onClick={() => setShowAuth(true)}>
            Đăng nhập
          </button>
        )}
      </div>
    </header>
  );
};

export default Navbar;
