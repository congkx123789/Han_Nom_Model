/* eslint-disable no-unused-vars */
import React from 'react';
import {
  LayoutDashboard,
  UploadCloud,
  Database,
  BookOpen,
  ShieldCheck,
  MessageSquare,
  ArrowLeftRight,
} from 'lucide-react';

const SidebarItem = ({ icon: Icon, label, active, onClick, collapsed }) => (
  <button
    onClick={onClick}
    className={`sidebar-item ${active ? 'active' : ''} ${collapsed ? 'collapsed' : ''}`}
    title={label}
  >
    <Icon size={18} />
    {!collapsed && <span>{label}</span>}
  </button>
);

const Sidebar = ({ activeTab, setActiveTab, user, setShowAuth }) => {
  const [collapsed, setCollapsed] = React.useState(false);

  const menuItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'workspace', label: 'Workspace', icon: UploadCloud },
    { id: 'archive', label: 'Archive/Delta Lake', icon: Database },
    { id: 'dictionary', label: 'Dictionary/RAG', icon: BookOpen },
    { id: 'chatbot', label: 'AI Chatbot', icon: MessageSquare },
    { id: 'admin', label: 'Admin Settings', icon: ShieldCheck },
  ];

  return (
    <aside className={`left-sidebar ${collapsed ? 'collapsed' : ''}`}>
      <div className="brand-mark">
        {!collapsed ? (
          <>
            <span className="brand-title">HÁN NÔM</span>
            <span className="brand-sub">HERITAGE</span>
          </>
        ) : (
          <span className="brand-sub">HN</span>
        )}
      </div>

      <nav className="sidebar-menu">
        {menuItems.map((item) => (
          <SidebarItem
            key={item.id}
            {...item}
            active={activeTab === item.id}
            onClick={() => setActiveTab(item.id)}
            collapsed={collapsed}
          />
        ))}
      </nav>

      <div className="sidebar-footer">
        <button
          onClick={() => setCollapsed((v) => !v)}
          className="collapse-btn"
          title="Thu gọn"
        >
          <ArrowLeftRight size={14} /> {!collapsed && 'Thu gọn'}
        </button>

        {!collapsed && (
          <button
            onClick={() => setActiveTab(activeTab === 'admin' ? 'dashboard' : 'admin')}
            className="admin-toggle"
          >
            CHUYỂN SANG ADMIN VIEW
          </button>
        )}

        <div className="profile-card" onClick={() => !user && setShowAuth(true)} style={{ cursor: user ? 'default' : 'pointer' }}>
          <div className="avatar">{user ? user.substring(0, 2).toUpperCase() : '??'}</div>
          {!collapsed && (
            <div>
              <div className="profile-name">{user ? user : 'Chưa đăng nhập'}</div>
              <div className="profile-role">{user ? 'Researcher' : 'Khách'}</div>
            </div>
          )}
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
