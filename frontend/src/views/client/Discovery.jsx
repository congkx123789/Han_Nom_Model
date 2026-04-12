import React from 'react';
import {
    Search as SearchIcon,
    Filter,
    Book,
    Clock,
    User,
    ChevronRight,
    ExternalLink,
    BookOpen
} from 'lucide-react';

const SearchResultItem = ({ title, era, author, snippet, type, accuracy }) => (
    <div className="scholarly-card p-4 hover:border-cinnabar/30 transition-all cursor-pointer group">
        <div className="flex gap-4">
            <div className="w-20 h-24 bg-gray-100 dark:bg-white/5 rounded border border-black/5 flex items-center justify-center shrink-0">
                <Book size={32} className="text-gray-300" />
            </div>
            <div className="flex-1 flex flex-col gap-1">
                <div className="flex justify-between items-start">
                    <h3 className="text-base font-bold italic group-hover:text-cinnabar transition-colors">{title}</h3>
                    <span className="text-[10px] font-bold text-gray-400 bg-gray-50 dark:bg-white/5 px-1.5 py-0.5 rounded">{type}</span>
                </div>
                <div className="flex gap-4 text-[10px] font-bold text-gray-500 uppercase tracking-tighter">
                    <span className="flex items-center gap-1"><User size={10} /> {author}</span>
                    <span className="flex items-center gap-1"><Clock size={10} /> {era}</span>
                    <span className="text-cinnabar">OCR: {accuracy}%</span>
                </div>
                <p className="text-xs text-gray-500 line-clamp-2 mt-1 leading-relaxed">
                    ... {snippet.split(' ').map((word, i) => i > 5 && i < 10 ? <b key={i} className="text-cinnabar font-bold">{word} </b> : word + ' ')} ...
                </p>
                <div className="flex gap-2 mt-2">
                    <button className="text-[9px] font-bold text-indigo hover:underline flex items-center gap-1">
                        <BookOpen size={10} /> CHI TIẾT TÀI LIỆU
                    </button>
                    <button className="text-[9px] font-bold text-gray-400 hover:text-cinnabar flex items-center gap-1">
                        <ExternalLink size={10} /> XEM BẢN GỐC
                    </button>
                </div>
            </div>
        </div>
    </div>
);

const Discovery = () => {
    const results = [
        { title: 'Nam Quốc Sơn Hà', author: 'Lý Thường Kiệt', era: 'Nhà Lý', type: 'Bản dập', accuracy: 99.2, snippet: 'Nam quốc sơn hà Nam đế cư tiệt nhiên định phận tại thiên thư' },
        { title: 'Đại Việt Sử Ký Toàn Thư', author: 'Ngô Sĩ Liên', era: 'Nhà Lê', type: 'Mộc bản', accuracy: 98.5, snippet: 'Sử kỉ toàn thư chép từ thời Hồng Bàng đến thời Lê Thái Tổ' },
        { title: 'Truyện Kiều', author: 'Nguyễn Du', era: 'Nhà Nguyễn', type: 'Thư tịch', accuracy: 94.1, snippet: 'Trăm năm trong cõi người ta chữ tài chữ mệnh khéo là ghét nhau' },
        { title: 'Bản dập Bia tiến sĩ', author: 'Nhiều tác giả', era: 'Lê - Mạc', type: 'Bia đá', accuracy: 89.7, snippet: 'Hiền tài là nguyên khí quốc gia nguyên khí thịnh thì thế nước mạnh' },
    ];

    return (
        <div className="flex gap-6 animate-in fade-in duration-500 h-full">
            {/* Search Sidebar Filters */}
            <aside className="w-64 flex flex-col gap-6 shrink-0">
                <div className="flex flex-col gap-2">
                    <h3 className="text-[10px] font-bold uppercase tracking-widest text-gray-400">Bộ lọc di sản</h3>
                    <div className="h-0.5 w-8 bg-cinnabar"></div>
                </div>

                <div className="flex flex-col gap-4">
                    {[
                        { label: 'Loại hình', items: ['Tất cả', 'Bia đá', 'Mộc bản', 'Sắc phong', 'Thư tịch cổ'] },
                        { label: 'Triều đại', items: ['Lý', 'Trần', 'Hồ', 'Lê', 'Mạc', 'Tây Sơn', 'Nguyễn'] },
                        { label: 'Tình trạng OCR', items: ['Đã hoàn thành', 'Đang xử lý', 'Chưa số hóa'] },
                    ].map((filter, i) => (
                        <div key={i} className="flex flex-col gap-2">
                            <span className="text-[10px] font-bold text-gray-600 uppercase">{filter.label}</span>
                            <div className="flex flex-col gap-1">
                                {filter.items.map((item, j) => (
                                    <label key={j} className="flex items-center gap-2 text-xs text-gray-500 hover:text-cinnabar cursor-pointer py-0.5">
                                        <input type="checkbox" className="w-3 h-3 accent-cinnabar" defaultChecked={j === 0} />
                                        {item}
                                    </label>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            </aside>

            {/* Results Area */}
            <div className="flex-1 flex flex-col gap-4">
                <div className="flex justify-between items-center text-[10px] font-bold text-gray-400 uppercase tracking-widest">
                    <span>Tìm thấy {results.length} kết quả phù hợp</span>
                    <div className="flex gap-4">
                        <span className="cursor-pointer text-cinnabar">Độ liên quan</span>
                        <span className="cursor-pointer hover:text-gray-600">Niên đại</span>
                        <span className="cursor-pointer hover:text-gray-600">A-Z</span>
                    </div>
                </div>

                <div className="flex flex-col gap-4">
                    {results.map((item, i) => (
                        <SearchResultItem key={i} {...item} />
                    ))}
                </div>

                <div className="mt-auto py-6 flex justify-center gap-2">
                    {[1, 2, 3, '...', 12].map((p, i) => (
                        <button key={i} className={`w-8 h-8 rounded text-[10px] font-bold border transition-colors ${p === 1 ? 'bg-cinnabar text-white border-cinnabar' : 'border-gray-200 hover:border-cinnabar'}`}>
                            {p}
                        </button>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default Discovery;
