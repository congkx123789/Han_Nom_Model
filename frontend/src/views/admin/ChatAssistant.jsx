import React, { useMemo, useState } from 'react';
import {
  Bot,
  Database,
  Send,
  Loader2,
  ExternalLink,
  MessageCircle,
} from 'lucide-react';

const promptSuggestions = [
  "Giải nghĩa câu 'Nam quốc sơn hà Nam đế cư'",
  "Tóm tắt nội dung Truyện Kiều bản Nôm",
  "So sánh chữ Nôm và chữ Hán trong văn bia thời Lý",
];

const initialMessages = [
  {
    role: 'assistant',
    text: 'Xin chào, tôi là Han-Nom AI Assistant. Bạn có thể hỏi ý nghĩa văn bản, dịch nghĩa, hoặc tra cứu trực tiếp dữ liệu di sản.',
    citations: [
      { title: 'Delta Lake /nam-quoc-son-ha/ban-dap-ly', href: '#' },
      { title: 'Milvus /han-viet-dictionary/chu-nam', href: '#' },
    ],
  },
];

const ChatAssistant = () => {
  const [mode, setMode] = useState('bot');
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState(initialMessages);

  // Store dynamic context data from the backend to display on the side panel
  const [contextData, setContextData] = useState({
    char_lookup: { char: '南', hanviet: 'Nam', meaning: 'Phương Nam, quốc hiệu, định vị chủ quyền.' },
    structure: { radicals: '十 + 冂 + 一 + 丨', variants: '楠, 喃' },
    examples: [
      'Nam quốc sơn hà (thơ thần, thời Lý)',
      'Việt Nam quốc sử diễn ca',
      'Văn bia chùa thời Trần'
    ]
  });

  const canSend = useMemo(() => input.trim().length > 0 && !loading, [input, loading]);

  const handleSend = async () => {
    const text = input.trim();
    if (!text) return;

    setInput('');
    setLoading(true);

    const newMsgs = [...messages, { role: 'user', text }];
    setMessages(newMsgs);

    try {
      // Connect specifically to our newly created FastAPI Chat route
      const res = await fetch('http://localhost:8000/api/v1/chat/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: text, mode: mode })
      });

      if (!res.ok) {
        throw new Error(`API error: ${res.status}`);
      }

      const data = await res.json();

      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          text: data.text,
          citations: data.citations || [],
        },
      ]);

      // Update Context Panel if metadata is attached
      if (data.contextual_data) {
        setContextData(data.contextual_data);
      }

    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          text: 'Rất tiếc, đã có lỗi khi kết nối tới máy chủ AI (Port 8000). Vui lòng kiểm tra lại dịch vụ Backend.',
          citations: [],
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="chat-scholar-wrap">
      <div className="chat-title-row">
        <div>
          <h1>AI Chat & Tra cứu dữ liệu (Học giả)</h1>
          <p>Giao diện trợ lý nghiên cứu Hán Nôm: hỏi bot hoặc truy vấn trực tiếp cơ sở dữ liệu.</p>
        </div>

        <div className="chat-actions">
          <button className={mode === 'bot' ? 'active' : ''} onClick={() => setMode('bot')}>
            <Bot size={15} /> Ask Bot
          </button>
          <button className={mode === 'db' ? 'active' : ''} onClick={() => setMode('db')}>
            <Database size={15} /> Query DB
          </button>
        </div>
      </div>

      <div className="prompt-carousel">
        {promptSuggestions.map((label) => (
          <button key={label} onClick={() => setInput(label)}>
            <span>{label}</span>
          </button>
        ))}
      </div>

      <div className="chat-layout-grid">
        <div className="chat-thread-panel">
          <div className="thread-head">
            <div>
              <strong>Threads</strong>
              <span>Scholar Session • Active</span>
            </div>
          </div>

          <div className="thread-body">
            {messages.map((msg, idx) => (
              <article key={`${msg.role}-${idx}`} className={`chat-bubble ${msg.role}`}>
                <div className="bubble-meta">
                  {msg.role === 'assistant' ? 'Hán-Nôm Heritage AI' : 'Scholar'}
                </div>
                <p>{msg.text}</p>

                {msg.role === 'assistant' && msg.citations?.length > 0 && (
                  <div className="bubble-citations">
                    {msg.citations.map((c) => (
                      <a key={c.title} href={c.href}>
                        <ExternalLink size={12} /> {c.title}
                      </a>
                    ))}
                  </div>
                )}
              </article>
            ))}

            {loading && (
              <article className="chat-bubble assistant loading-bubble">
                <Loader2 size={14} className="spin" /> Đang tổng hợp dữ liệu từ Backend...
              </article>
            )}
          </div>

          <div className="chat-composer" style={{ position: 'relative' }}>
            {input === '/' && (
              <div className="slash-dropdown">
                <button className="slash-item" onClick={() => setInput('/dich ')}>
                  <span className="cmd-tag">/dich</span>
                  <span className="cmd-desc">Dịch đoạn văn bản Hán Nôm</span>
                </button>
                <button className="slash-item" onClick={() => setInput('/tracuu ')}>
                  <span className="cmd-tag">/tracuu</span>
                  <span className="cmd-desc">Tìm chữ trong kho di sản</span>
                </button>
                <button className="slash-item" onClick={() => setInput('/tomtat ')}>
                  <span className="cmd-tag">/tomtat</span>
                  <span className="cmd-desc">Tóm tắt bối cảnh lịch sử</span>
                </button>
              </div>
            )}
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Nhập truy vấn học thuật, hoặc gõ / để gọi menu..."
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            />
            <button className="send-btn" disabled={!canSend} onClick={handleSend}>
              {loading ? <Loader2 size={14} className="spin" /> : <Send size={14} />} Gửi
            </button>
          </div>
        </div>

        <aside className="context-panel">
          <header>
            <h3>Contextual Side Panel</h3>
            <span>Milvus • Delta Lake</span>
          </header>

          <div className="context-block">
            <h4>Từ đang tra cứu</h4>
            <p className="nom-char">{contextData.char_lookup.char}</p>
            <p>Âm Hán Việt: <strong>{contextData.char_lookup.hanviet}</strong></p>
            <p>Nghĩa: {contextData.char_lookup.meaning}</p>
          </div>

          <div className="context-block">
            <h4>Cấu tạo chữ</h4>
            <p>Bộ thủ: {contextData.structure.radicals}</p>
            <p>Liên hệ dị thể: {contextData.structure.variants}</p>
          </div>

          <div className="context-block">
            <h4>Ví dụ sử dụng</h4>
            <ul>
              {contextData.examples.map((ex, i) => (
                <li key={i}>{ex}</li>
              ))}
            </ul>
          </div>
        </aside>
      </div>

      <button className="floating-ask-ai">
        <MessageCircle size={16} />
        <span>Ask AI</span>
      </button>
    </section>
  );
};

export default ChatAssistant;
