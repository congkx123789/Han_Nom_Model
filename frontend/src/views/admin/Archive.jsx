import React from 'react';
import { Database, Filter, MoreVertical, Layers, HardDrive, Zap, ChevronDown, CheckCircle2, Clock, Trash2, RotateCcw, Eye } from 'lucide-react';

const Archive = () => {
    const documents = [
        { id: 'HN-001', name: 'Đại Việt Sử Ký Toàn Thư', type: 'Mộc bản', progress: 100, owner: 'Lê Văn Hưu', date: '2026-04-01' },
        { id: 'HN-002', name: 'Bình Ngô Đại Cáo', type: 'Sắc phong', progress: 75, owner: 'Nguyễn Trãi', date: '2026-04-05' },
        { id: 'HN-003', name: 'Sơn Cư Tạp Thuật', type: 'Thư tịch', progress: 12, owner: 'Hàn Thuyên', date: '2026-04-07' },
        { id: 'HN-004', name: 'Văn bia Quốc Tử Giám', type: 'Bia đá', progress: 100, owner: 'Nhà Lê', date: '2026-03-20' },
    ];

    return (
        <div className="flex flex-col gap-6 animate-in slide-in-from-right-4 duration-500 h-full">
            {/* Header & Action Bar */}
            <div className="flex justify-between items-end">
                <div className="flex flex-col gap-1">
                    <h2 className="text-2xl font-bold tracking-tight text-gray-900 dark:text-gray-100">Kho Lưu trữ Delta Lake</h2>
                    <p className="text-gray-500 text-sm">Quản lý và truy xuất 1.284 văn bản di sản đã được số hóa.</p>
                </div>

                <div className="flex items-center gap-3">
                    <div className="flex items-center gap-2 mr-2">
                        <span className="text-[10px] font-bold text-gray-400 uppercase tracking-widest bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded">ĐÃ CHỌN: 02</span>
                    </div>

                    <button className="px-3 py-2 bg-gray-100 dark:bg-zinc-800 border border-gray-200 dark:border-gray-700 text-xs font-bold rounded flex items-center gap-2 hover:bg-gray-200 dark:hover:bg-zinc-700 transition-colors text-gray-700 dark:text-gray-300">
                        <Filter size={14} /> BỘ LỌC <ChevronDown size={14} className="opacity-50" />
                    </button>

                    <button className="px-3 py-2 bg-gray-100 dark:bg-zinc-800 border border-gray-200 dark:border-gray-700 text-[10px] font-bold rounded flex items-center gap-2 hover:bg-gray-200 dark:hover:bg-zinc-700 transition-colors uppercase tracking-widest text-gray-700 dark:text-gray-300">
                        TÁC VỤ <ChevronDown size={14} className="opacity-50" />
                    </button>

                    <button className="px-4 py-2 bg-cinnabar text-white text-xs font-bold rounded flex items-center gap-2 hover:bg-red-700 transition-all shadow-lg shadow-red-900/20">
                        <Zap size={14} className="fill-white" /> RUN SPARK JOB
                    </button>
                </div>
            </div>

            {/* Rich Data Grid Container */}
            <div className="scholarly-card flex-1 overflow-hidden flex flex-col bg-white dark:bg-zinc-950 border border-gray-200 dark:border-gray-800">
                <div className="p-4 border-b border-gray-100 dark:border-gray-800 flex gap-8 items-center bg-gray-50/50 dark:bg-black/20">
                    <div className="flex items-center gap-2 text-indigo-500">
                        <HardDrive size={16} />
                        <span className="text-xs font-bold uppercase tracking-widest">Active Storage: MinIO Cluster A</span>
                    </div>
                    <div className="flex-1"></div>
                    <span className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">HIỂN THỊ 1-10 TRÊN 1,284</span>
                </div>

                <div className="overflow-auto flex-1 custom-scrollbar">
                    <table className="w-full text-left text-sm border-separate border-spacing-0">
                        <thead className="sticky top-0 bg-gray-50 dark:bg-gray-900 z-10 shadow-[0_1px_0_rgba(255,255,255,0.05)] border-b border-gray-800">
                            <tr className="text-[10px] font-bold uppercase text-gray-500 tracking-widest">
                                <th className="p-4 border-b border-gray-200 dark:border-gray-800">Tên tài liệu</th>
                                <th className="p-4 border-b border-gray-200 dark:border-gray-800">Mã ID</th>
                                <th className="p-4 border-b border-gray-200 dark:border-gray-800">Loại hình</th>
                                <th className="p-4 border-b border-gray-200 dark:border-gray-800 min-w-[200px]">Tiến trình OCR</th>
                                <th className="p-4 border-b border-gray-200 dark:border-gray-800">Cập nhật</th>
                                <th className="p-4 border-b border-gray-200 dark:border-gray-800 text-center">Tác vụ</th>
                            </tr>
                        </thead>
                        <tbody>
                            {documents.map((doc, i) => (
                                <tr key={i} className="group hover:bg-gray-50 dark:hover:bg-white/5 transition-colors">
                                    <td className="p-4 border-b border-gray-100 dark:border-gray-800 font-semibold text-gray-900 dark:text-gray-100">{doc.name}</td>
                                    <td className="p-4 border-b border-gray-100 dark:border-gray-800 font-mono text-[11px] text-gray-500 dark:text-gray-400">{doc.id}</td>
                                    <td className="p-4 border-b border-gray-100 dark:border-gray-800">
                                        <span className="px-2.5 py-1 bg-gray-100 dark:bg-gray-800/80 border border-gray-200 dark:border-gray-700 rounded text-[10px] font-bold text-gray-700 dark:text-gray-300">
                                            {doc.type}
                                        </span>
                                    </td>
                                    <td className="p-4 border-b border-gray-100 dark:border-gray-800">
                                        <div className="flex items-center gap-3">
                                            {doc.progress === 100 ? (
                                                <span className="flex items-center gap-1.5 px-2 py-0.5 bg-green-500/10 text-green-600 dark:text-green-500 border border-green-500/20 rounded-full text-[10px] font-bold uppercase tracking-wider">
                                                    <CheckCircle2 size={12} /> Completed
                                                </span>
                                            ) : (
                                                <span className="flex items-center gap-1.5 px-2 py-0.5 bg-yellow-500/10 text-yellow-600 dark:text-yellow-500 border border-yellow-500/20 rounded-full text-[10px] font-bold uppercase tracking-wider">
                                                    <Clock size={12} /> Processing
                                                </span>
                                            )}

                                            <div className="flex-1 h-1.5 bg-gray-200 dark:bg-gray-800 rounded-full overflow-hidden">
                                                <div
                                                    className={`h-full transition-all duration-1000 ${doc.progress === 100 ? 'bg-green-500' : 'bg-yellow-500'}`}
                                                    style={{ width: `${doc.progress}%` }}
                                                ></div>
                                            </div>
                                        </div>
                                    </td>
                                    <td className="p-4 border-b border-gray-100 dark:border-gray-800 text-xs text-gray-500 dark:text-gray-400 font-mono">
                                        {doc.date}
                                    </td>
                                    <td className="p-4 border-b border-gray-100 dark:border-gray-800 text-center relative">
                                        <div className="relative inline-block text-left">
                                            <button className="p-2 hover:bg-gray-200 dark:hover:bg-gray-800 rounded text-gray-500 dark:text-gray-400 transition-colors peer">
                                                <MoreVertical size={16} />
                                            </button>

                                            {/* Popover Menu */}
                                            <div className="hidden peer-hover:block hover:block absolute right-6 top-0 w-36 bg-white dark:bg-zinc-900 border border-gray-200 dark:border-gray-800 rounded-lg shadow-xl z-50 overflow-hidden">
                                                <div className="flex flex-col text-xs font-semibold">
                                                    <button className="flex items-center gap-2 px-3 py-2 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800 w-full text-left">
                                                        <Eye size={14} /> Xem chi tiết
                                                    </button>
                                                    <button className="flex items-center gap-2 px-3 py-2 text-indigo-600 dark:text-indigo-400 hover:bg-gray-50 dark:hover:bg-gray-800 w-full text-left">
                                                        <RotateCcw size={14} /> Chạy lại OCR
                                                    </button>
                                                    <div className="h-[1px] bg-gray-200 dark:bg-gray-800 w-full"></div>
                                                    <button className="flex items-center gap-2 px-3 py-2 text-red-600 dark:text-red-500 hover:bg-red-50 dark:hover:bg-red-500/10 w-full text-left">
                                                        <Trash2 size={14} /> Xóa tài liệu
                                                    </button>
                                                </div>
                                            </div>
                                        </div>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
};

export default Archive;
