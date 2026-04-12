import React, { useState, useEffect, useRef } from 'react';
import { Search, MessageSquare, Copy, X } from 'lucide-react';

const FloatingMenu = ({ onAction }) => {
    const [position, setPosition] = useState({ top: 0, left: 0, visible: false });
    const [selectedText, setSelectedText] = useState('');
    const menuRef = useRef(null);

    useEffect(() => {
        const handleMouseUp = () => {
            const selection = window.getSelection();
            const text = selection.toString().trim();

            if (text && text.length > 0) {
                const range = selection.getRangeAt(0);
                const rect = range.getBoundingClientRect();

                setSelectedText(text);
                setPosition({
                    top: rect.top + window.scrollY - 45,
                    left: rect.left + window.scrollX + rect.width / 2,
                    visible: true,
                });
            } else {
                // Only hide if not clicking inside the menu
                setTimeout(() => {
                    if (!menuRef.current?.contains(document.activeElement)) {
                        // setPosition(prev => ({ ...prev, visible: false }));
                    }
                }, 100);
            }
        };

        const handleMouseDown = (e) => {
            if (menuRef.current && !menuRef.current.contains(e.target)) {
                setPosition(prev => ({ ...prev, visible: false }));
            }
        }

        document.addEventListener('mouseup', handleMouseUp);
        document.addEventListener('mousedown', handleMouseDown);
        return () => {
            document.removeEventListener('mouseup', handleMouseUp);
            document.removeEventListener('mousedown', handleMouseDown);
        };
    }, []);

    if (!position.visible) return null;

    return (
        <div
            ref={menuRef}
            className="floating-selection-menu"
            style={{
                position: 'absolute',
                top: position.top,
                left: position.left,
                transform: 'translateX(-50%)',
                zIndex: 9999,
            }}
        >
            <div className="menu-inner">
                <button title="Tra từ điển" onClick={() => { onAction('lookup', selectedText); setPosition(p => ({ ...p, visible: false })); }}>
                    <Search size={14} />
                </button>
                <button title="Hỏi AI" onClick={() => { onAction('ask', selectedText); setPosition(p => ({ ...p, visible: false })); }}>
                    <MessageSquare size={14} />
                </button>
                <button title="Copy" onClick={() => { navigator.clipboard.writeText(selectedText); setPosition(p => ({ ...p, visible: false })); }}>
                    <Copy size={14} />
                </button>
                <div className="divider"></div>
                <button className="close-btn" onClick={() => setPosition(p => ({ ...p, visible: false }))}>
                    <X size={12} />
                </button>
            </div>
        </div>
    );
};

export default FloatingMenu;
