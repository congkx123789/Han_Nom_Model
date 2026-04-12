import React, { useState } from 'react';
import { X, User, Lock, Mail } from 'lucide-react';

const AuthModal = ({ isOpen, onClose, onLogin }) => {
    const [isLogin, setIsLogin] = useState(true);
    const [formData, setFormData] = useState({ username: '', email: '', password: '' });

    if (!isOpen) return null;

    const handleSubmit = (e) => {
        e.preventDefault();
        // Dummy login logic
        if (formData.username.trim() === '') {
            onLogin(formData.email.split('@')[0] || 'Scholar');
        } else {
            onLogin(formData.username);
        }
    };

    return (
        <div className="auth-modal-overlay" onClick={onClose}>
            <div className="auth-modal-content" onClick={e => e.stopPropagation()}>
                <button className="auth-close-btn" onClick={onClose}>
                    <X size={20} />
                </button>

                <h2 className="auth-title">{isLogin ? 'Đăng nhập vào hệ thống' : 'Tạo tài khoản học giả'}</h2>
                <p className="auth-subtitle">
                    {isLogin
                        ? 'Tiếp tục công việc số hóa và tra cứu Hán Nôm của bạn.'
                        : 'Tham gia nền tảng chia sẻ và lưu trữ di sản Hán Nôm.'}
                </p>

                <form onSubmit={handleSubmit} className="auth-form">
                    {!isLogin && (
                        <div className="input-group">
                            <User size={16} className="input-icon" />
                            <input
                                type="text"
                                placeholder="Họ và tên hoặc Bút danh"
                                value={formData.username}
                                onChange={e => setFormData({ ...formData, username: e.target.value })}
                                required={!isLogin}
                            />
                        </div>
                    )}

                    <div className="input-group">
                        <Mail size={16} className="input-icon" />
                        <input
                            type="text"
                            placeholder="Email hoặc Tên đăng nhập"
                            value={formData.email}
                            onChange={e => setFormData({ ...formData, email: e.target.value })}
                            required
                        />
                    </div>

                    <div className="input-group">
                        <Lock size={16} className="input-icon" />
                        <input
                            type="password"
                            placeholder="Mật khẩu"
                            value={formData.password}
                            onChange={e => setFormData({ ...formData, password: e.target.value })}
                            required
                        />
                    </div>

                    <button type="submit" className="auth-submit-btn">
                        {isLogin ? 'Đăng nhập' : 'Đăng ký'}
                    </button>
                </form>

                <div className="auth-switch">
                    <span>{isLogin ? 'Chưa có tài khoản?' : 'Đã có tài khoản?'}</span>
                    <button type="button" onClick={() => setIsLogin(!isLogin)} className="auth-switch-btn">
                        {isLogin ? 'Đăng ký ngay' : 'Đăng nhập'}
                    </button>
                </div>
            </div>
        </div>
    );
};

export default AuthModal;
