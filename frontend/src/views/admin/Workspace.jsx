import React, { useState } from 'react';
import {
    ZoomIn,
    ZoomOut,
    RotateCw,
    Maximize2,
    Save,
    Shield,
    Brain,
    BookOpen,
    Image as ImageIcon,
    Database,
    Search,
    MessageSquare,
    Copy,
    Info
} from 'lucide-react';
import FloatingMenu from '../../components/common/FloatingMenu';


const Workspace = () => {
    const [activeChar, setActiveChar] = useState(null);

    return (
        <div className="flex flex-col h-full gap-4 animate-in slide-in-from-bottom-4 duration-500">
            {/* Top Toolbar */}
            <div className="flex justify-between items-center bg-zinc-50 dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 p-3 rounded-lg shadow-sm">
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-red-600/10 rounded">
                        <BookOpen className="text-cinnabar" size={20} />
                    </div>
                    <div>
                        <h2 className="text-sm font-bold uppercase tracking-widest text-zinc-900 dark:text-zinc-100">Nam Quốc Sơn Hà</h2>
                        <p className="text-[10px] text-zinc-500 font-mono">TÀI LIỆU SỐ #4829</p>
                    </div>
                </div>

                <div className="flex gap-2">
                    <button className="px-4 py-2 bg-zinc-100 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 text-zinc-800 dark:text-zinc-300 text-xs font-bold rounded flex items-center gap-2 hover:bg-zinc-200 dark:hover:bg-zinc-700 transition-colors shadow-sm">
                        <Save size={14} /> LƯU BẢN THẢO
                    </button>
                    <button className="px-4 py-2 bg-green-600 text-white text-xs font-bold rounded flex items-center gap-2 hover:bg-green-700 transition-colors shadow-lg shadow-green-900/20">
                        <Shield size={14} /> DUYỆT (HỌC GIẢ)
                    </button>
                </div>
            </div>

            {/* Split Pane 50/50 Layout */}
            <div className="flex-1 min-h-0 flex gap-6">

                {/* 50% LEFT: Image Viewer Canvas */}
                <div className="w-1/2 bg-zinc-100 dark:bg-zinc-950 border border-zinc-300 dark:border-zinc-800 rounded-xl flex flex-col relative overflow-hidden group shadow-inner">
                    {/* Floating Toolbar */}
                    <div className="absolute top-4 right-4 z-10 flex gap-1 bg-white/90 dark:bg-zinc-900/90 backdrop-blur-md p-1.5 rounded-lg border border-zinc-300 dark:border-zinc-700 shadow-lg opacity-0 group-hover:opacity-100 transition-all duration-300 translate-y-[-10px] group-hover:translate-y-0">
                        {[ZoomIn, ZoomOut, RotateCw, Maximize2].map((Icon, i) => (
                            <button key={i} className="p-2 hover:bg-zinc-200 dark:hover:bg-zinc-800 rounded text-zinc-700 dark:text-zinc-300 transition-colors">
                                <Icon size={16} />
                            </button>
                        ))}
                    </div>

                    <div className="flex-1 flex items-center justify-center p-4 overflow-auto">
                        <div className="relative border-4 border-double border-zinc-400 dark:border-zinc-700 p-2 shadow-2xl bg-[#f4ebd0] dark:bg-[#1a1714]">
                            <img
                                src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Nam_Quốc_Sơn_Hà.jpg/450px-Nam_Quốc_Sơn_Hà.jpg"
                                alt="Bản scan Nam Quốc Sơn Hà"
                                className="max-h-[70vh] mix-blend-multiply dark:mix-blend-luminosity opacity-90"
                            />
                        </div>
                    </div>
                </div>

                {/* 50% RIGHT: Data Extraction Panel */}
                <div className="w-1/2 flex flex-col gap-4 overflow-hidden">
                    {/* Specification Badges */}
                    <div className="flex gap-2 items-center flex-wrap">
                        <span className="flex items-center gap-1.5 px-2.5 py-1 bg-zinc-100 dark:bg-zinc-900 border border-zinc-300 dark:border-zinc-800 rounded-md text-[10px] font-bold uppercase text-zinc-600 dark:text-zinc-400">
                            <ImageIcon size={12} /> RES: 4096 x 8192px
                        </span>
                        <span className="flex items-center gap-1.5 px-2.5 py-1 bg-zinc-100 dark:bg-zinc-900 border border-zinc-300 dark:border-zinc-800 rounded-md text-[10px] font-bold uppercase text-zinc-600 dark:text-zinc-400">
                            <Database size={12} /> Storage: MinIO
                        </span>
                        <span className="px-2.5 py-1 bg-green-500/10 border border-green-500/20 text-green-600 dark:text-green-500 rounded-md text-[10px] font-bold uppercase tracking-widest">
                            ✓ Hoàn thành
                        </span>
                    </div>

                    {/* Data Table */}
                    <div className="flex-1 bg-white dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 rounded-xl overflow-hidden flex flex-col shadow-sm">
                        <div className="p-4 border-b border-zinc-200 dark:border-zinc-800 flex gap-6 text-[11px] font-bold uppercase text-zinc-500 dark:text-zinc-400 bg-zinc-50 dark:bg-zinc-900/50">
                            <span className="text-cinnabar border-b-2 border-cinnabar pb-3 -mb-4">Kết Quả Bóc Tách Hán Nôm</span>
                            <span className="hover:text-zinc-900 dark:hover:text-zinc-200 cursor-pointer transition-colors">Metadata</span>
                        </div>

                        <div className="flex-1 overflow-y-auto custom-scrollbar">
                            <table className="w-full text-sm border-collapse">
                                <thead className="bg-zinc-100 dark:bg-zinc-900 text-[10px] uppercase text-zinc-500 dark:text-zinc-400 font-bold sticky top-0 z-10 border-b border-zinc-200 dark:border-zinc-800">
                                    <tr>
                                        <th className="p-3 text-center border-r border-zinc-200 dark:border-zinc-800 w-1/3">Hán Nôm</th>
                                        <th className="p-3 text-center border-r border-zinc-200 dark:border-zinc-800 w-1/3">Hán Việt</th>
                                        <th className="p-3 text-center w-1/3">Tiếng Việt</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {[
                                        { nom: '南', hv: 'Nam', vi: 'Phương Nam' },
                                        { nom: '國', hv: 'Quốc', vi: 'Đất nước' },
                                        { nom: '山', hv: 'Sơn', vi: 'Núi' },
                                        { nom: '河', hv: 'Hà', vi: 'Sông' },
                                        { nom: '南', hv: 'Nam', vi: 'Phương Nam' },
                                        { nom: '帝', hv: 'Đế', vi: 'Vua' },
                                        { nom: '居', hv: 'Cư', vi: 'Ở' },
                                    ].map((row, i) => (
                                        <tr
                                            key={i}
                                            onClick={() => setActiveChar(row)}
                                            className={`border-b border-zinc-100 dark:border-zinc-800/50 cursor-pointer transition-colors
                                                ${activeChar?.nom === row.nom
                                                    ? 'bg-red-50 dark:bg-red-900/10 ring-1 ring-inset ring-cinnabar/30'
                                                    : 'hover:bg-zinc-50 dark:hover:bg-zinc-900/70'}`}
                                        >
                                            <td className="p-4 text-center border-r border-zinc-100 dark:border-zinc-800/50">
                                                <span className="font-serif text-3xl font-normal text-zinc-900 dark:text-zinc-100 drop-shadow-sm">{row.nom}</span>
                                            </td>
                                            <td className="p-4 text-center font-semibold text-zinc-700 dark:text-zinc-300 border-r border-zinc-100 dark:border-zinc-800/50">
                                                {row.hv}
                                            </td>
                                            <td className="p-4 text-center text-zinc-500 dark:text-zinc-400">
                                                {row.vi}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    {/* AI Info Panel */}
                    {activeChar && (
                        <div className="h-64 bg-slate-50 dark:bg-zinc-950 border border-sky-200 dark:border-sky-900/50 p-5 rounded-xl shadow-md flex flex-col gap-4 animate-in slide-in-from-right-4 duration-300 overflow-hidden relative">
                            <div className="absolute -top-10 -right-10 w-32 h-32 bg-sky-500/10 dark:bg-sky-500/15 rounded-full blur-3xl pointer-events-none"></div>

                            <div className="flex justify-between items-start z-10">
                                <div className="flex gap-2 items-center">
                                    <div className="p-1.5 bg-sky-500/20 rounded text-sky-600 dark:text-sky-400">
                                        <Brain size={14} />
                                    </div>
                                    <span className="text-[10px] font-bold uppercase tracking-widest text-sky-700 dark:text-sky-400">AI Analysis (Qwen2.5-VL / Milvus)</span>
                                </div>
                                <button onClick={() => setActiveChar(null)} className="text-zinc-400 hover:text-zinc-800 dark:hover:text-zinc-200 transition-colors p-1">✕</button>
                            </div>

                            <div className="flex flex-col gap-3 z-10">
                                <div className="flex items-baseline gap-3">
                                    <span className="text-4xl font-serif text-zinc-900 dark:text-zinc-50">{activeChar.nom}</span>
                                    <span className="text-lg font-bold text-zinc-700 dark:text-zinc-300 border-l-2 border-zinc-300 dark:border-zinc-700 pl-3">{activeChar.hv}</span>
                                </div>
                                <p className="text-sm text-zinc-600 dark:text-zinc-400 leading-relaxed">
                                    Chữ <span className="font-serif font-bold text-zinc-900 dark:text-zinc-200">"{activeChar.nom}"</span> mang ý nghĩa là <strong className="text-zinc-800 dark:text-zinc-200">{activeChar.vi.toLowerCase()}</strong>.
                                    Đây là một ký tự biểu ý thuộc ngôn ngữ Hán Nôm cổ, thường xuất hiện trong các ngữ cảnh trang trọng.
                                </p>
                                <div className="flex gap-2 mt-2">
                                    <span className="text-[10px] bg-sky-100 dark:bg-sky-900/30 text-sky-700 dark:text-sky-400 px-2 py-1 border border-sky-200 dark:border-sky-800/50 rounded font-bold tracking-widest uppercase">NGỮ CẢNH: QUÂN CHỦ</span>
                                    <span className="text-[10px] bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 px-2 py-1 border border-red-200 dark:border-red-800/50 rounded font-bold tracking-widest uppercase">TẦN SUẤT: CAO</span>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
            <FloatingMenu
                onAction={(type, text) => {
                    if (type === 'lookup') {
                        // Trình kích hoạt tra cứu nhanh (Simulated)
                        alert(`Đang tra cứu từ điển cho: ${text}`);
                    } else if (type === 'ask') {
                        // Chuyển sang màn hình chat với câu hỏi tự động
                        window.location.hash = `#/chat?q=${encodeURIComponent(text)}`;
                    }
                }}
            />
        </div>
    );
};


export default Workspace;
