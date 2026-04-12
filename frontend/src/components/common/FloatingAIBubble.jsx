import React, { useState } from 'react';
import { MessageSquare, X, Send, Sparkles } from 'lucide-react';

const FloatingAIBubble = () => {
    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState([
        { role: 'assistant', text: 'Chào bạn! Tôi có thể giúp gì cho bạn về di sản Hán Nôm hôm nay?' }
    ]);
    const [input, setInput] = useState('');

    const handleSend = () => {
        if (!input.trim()) return;
        setMessages([...messages, { role: 'user', text: input }]);
        setInput('');

        // Simulating AI response logic (would call Backend RAG in real use)
        setTimeout(() => {
            setMessages(prev => [...prev, {
                role: 'assistant',
                text: 'Đây là phiên bản demo của trợ lý thông minh. Trong thực tế, tôi sẽ truy vấn Milvus và Qwen để trả lời bạn.'
            }]);
        }, 800);
    };

    return (
        <div className="discovery-ai-container">
            {isOpen ? (
                <div className="mini-chat-window animate-in slide-in-from-bottom-5 fade-in duration-300">
                    <div className="mini-chat-header">
                        <div className="flex items-center gap-2">
                            <Sparkles size={16} className="text-accent" />
                            <span className="font-bold text-sm">Hán Nôm AI Assistant</span>
                        </div>
                        <button onClick={() => setIsOpen(false)} className="close-mini-chat">
                            <X size={16} />
                        </button>
                    </div>

                    <div className="mini-chat-body">
                        {messages.map((m, i) => (
                            <div key={i} className={`mini-msg ${m.role}`}>
                                {m.text}
                            </div>
                        ))}
                    </div>

                    <div className="mini-chat-footer">
                        <input
                            type="text"
                            placeholder="Nhập câu hỏi..."
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                        />
                        <button onClick={handleSend} className="send-mini-btn">
                            <Send size={16} />
                        </button>
                    </div>
                </div>
            ) : (
                <button
                    className="ai-floating-trigger group"
                    onClick={() => setIsOpen(true)}
                    title="Hỏi trợ lý AI"
                >
                    <MessageSquare size={24} className="group-hover:scale-110 transition-transform" />
                    <div className="bubble-tooltip">Hỏi AI về di sản này</div>
                </button>
            )}
        </div>
    );
};

export default FloatingAIBubble;
