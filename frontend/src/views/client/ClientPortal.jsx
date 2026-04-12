import React, { useState } from 'react';
import {
    Search,
    ArrowRight,
    Image as ImageIcon,
    BookOpen,
    History,
    Sparkles,
    ChevronDown
} from 'lucide-react';

const HeritageCard = ({ img, title, category, year, description }) => (
    <div className="heritage-card group">
        <div className="heritage-card-image">
            {img ? (
                <img src={img} alt={title} />
            ) : (
                <div className="heritage-placeholder">
                    <ImageIcon size={48} strokeWidth={1} />
                </div>
            )}
            <div className="heritage-card-overlay">
                <button className="view-btn">KHÁM PHÁ <ArrowRight size={14} /></button>
            </div>
        </div>
        <div className="heritage-card-info">
            <div className="heritage-card-meta">
                <span className="category">{category}</span>
                <span className="year">{year}</span>
            </div>
            <h3 className="heritage-card-title">{title}</h3>
            <p className="heritage-card-desc">{description}</p>
        </div>
    </div>
);

const ClientPortal = ({ onSearch }) => {
    const [searchQuery, setSearchQuery] = useState('');

    const featuredArtifacts = [
        {
            title: "Nam Quốc Sơn Hà",
            category: "Thơ Văn",
            year: "Thế kỷ XI",
            description: "Bản tuyên ngôn độc lập đầu tiên của dân tộc Việt Nam, khẳng định chủ quyền lãnh thổ."
        },
        {
            title: "Đại Việt Sử Ký Toàn Thư",
            category: "Lịch sử",
            year: "Thế kỷ XV",
            description: "Bộ quốc sử viết bằng chữ Hán chép từ thời Hồng Bàng đến thời Lê Thái Tổ."
        },
        {
            title: "Truyện Kiều (Bản Nôm)",
            category: "Văn học",
            year: "Thế kỷ XIX",
            description: "Kiệt tác của Nguyễn Du, đỉnh cao của văn học chữ Nôm truyền tụng qua nhiều đời."
        },
        {
            title: "Sắc phong triều Nguyễn",
            category: "Hành chính",
            year: "1802 - 1945",
            description: "Văn bản hành chính do triều đình ban tặng cho các cá nhân hoặc thần linh."
        }
    ];

    return (
        <div className="client-portal">
            {/* HERO SECTION */}
            <section className="client-hero">
                <div className="hero-content">
                    <div className="hero-badge">
                        <Sparkles size={12} /> NỀN TẢNG DI SẢN SỐ HÁN NÔM
                    </div>
                    <h1 className="hero-title">Giao thoa giữa Cổ điển và <span className="text-accent">Trí tuệ Nhân tạo</span></h1>
                    <p className="hero-subtitle">
                        Tìm kiếm, dịch thuật và phân tích hàng ngàn trang thư tịch cổ, mộc bản và bia đá thông qua công nghệ OCR & RAG tiên tiến.
                    </p>

                    <div className="hero-search-container">
                        <div className="hero-search-box">
                            <Search size={20} className="search-icon" />
                            <input
                                type="text"
                                placeholder="Hãy hỏi về một chữ Hán, một tác phẩm hoặc thời đại..."
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                onKeyDown={(e) => e.key === 'Enter' && onSearch(searchQuery)}
                            />
                            <button className="hero-search-btn" onClick={() => onSearch(searchQuery)}>
                                TÌM KIẾM
                            </button>
                        </div>
                        <div className="search-suggestions">
                            <span>Gợi ý:</span>
                            <button onClick={() => setSearchQuery('Chữ Nhẫn')}>Chữ Nhẫn</button>
                            <button onClick={() => setSearchQuery('Lý Thường Kiệt')}>Lý Thường Kiệt</button>
                            <button onClick={() => setSearchQuery('Bia tiến sĩ')}>Bia tiến sĩ</button>
                        </div>
                    </div>
                </div>

                <div className="scroll-indicator">
                    <span>Khám phá kho di sản</span>
                    <ChevronDown size={14} />
                </div>
            </section>

            {/* CURATED GALLERY */}
            <section className="client-gallery">
                <div className="section-head">
                    <h2 className="section-title">Di Sản Nổi Bật</h2>
                    <p className="section-desc">Những tài liệu quý hiếm đã được số hóa và phục dựng qua AI.</p>
                </div>

                <div className="heritage-grid">
                    {featuredArtifacts.map((art, i) => (
                        <HeritageCard key={i} {...art} />
                    ))}
                </div>

                <div className="view-more-container">
                    <button className="outline-btn">XEM TẤT CẢ KHO TÀNG <History size={16} /></button>
                </div>
            </section>

            {/* AI CAPABILITIES PREVIEW */}
            <section className="client-features">
                <div className="feature-item">
                    <div className="feature-icon"><BookOpen size={24} /></div>
                    <h3>Tra cứu Chính xác</h3>
                    <p>Kết nối trực tiếp với 58,000 mục từ điển và văn bản gốc Hán Nôm.</p>
                </div>
                <div className="feature-item">
                    <div className="feature-icon"><Sparkles size={24} /></div>
                    <h3>Dịch thuật Công nghệ</h3>
                    <p>Sử dụng Qwen 2.5-VL 3B để bóc tách ý nghĩa chuyên sâu của cổ văn.</p>
                </div>
            </section>
        </div>
    );
};

export default ClientPortal;
