import React from 'react';
import { Shield, Zap, Cpu, Activity, AlertCircle, Terminal, HardDrive } from 'lucide-react';

const Admin = () => {
    return (
        <div className="flex flex-col gap-6 animate-in zoom-in-95 duration-500">
            <div className="flex flex-col gap-1">
                <h2 className="text-2xl font-bold tracking-tight text-zinc-900 dark:text-zinc-100">Hệ thống & Hạ tầng</h2>
                <p className="text-zinc-500 text-sm">Giám sát tài nguyên GPU, Spark clusters và logs hệ thống thời gian thực.</p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* GPU Cluster Monitoring */}
                <div className="bg-white dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 rounded-xl p-5 flex flex-col gap-5 shadow-sm">
                    <div className="flex justify-between items-center">
                        <h3 className="text-xs font-bold uppercase tracking-widest flex items-center gap-2 text-zinc-800 dark:text-zinc-200">
                            <Cpu size={16} className="text-cinnabar" />
                            NVIDIA GPU CLUSTER
                        </h3>
                        <span className="text-[9px] bg-green-500/10 text-green-600 dark:text-green-500 border border-green-500/20 px-2 py-0.5 rounded-full font-bold">HEALTHY</span>
                    </div>

                    <div className="flex flex-col gap-5 flex-1 justify-center">
                        <div>
                            <div className="flex justify-between text-[11px] font-bold mb-2">
                                <span className="text-zinc-500 dark:text-zinc-400">A100 NODE 01 (RAG)</span>
                                <span className="flex gap-2">
                                    <span className="text-zinc-700 dark:text-zinc-300">84%</span>
                                    <span className="text-red-600 dark:text-red-500 animate-pulse">94°C</span>
                                </span>
                            </div>
                            <div className="h-2.5 w-full bg-zinc-100 dark:bg-zinc-800 rounded-full overflow-hidden">
                                <div className="h-full bg-red-500 w-[84%] relative">
                                    <div className="absolute inset-0 bg-white/20 w-full animate-[shimmer_2s_infinite]"></div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="flex justify-between text-[11px] font-bold mb-2">
                                <span className="text-zinc-500 dark:text-zinc-400">L40S NODE 02 (OCR)</span>
                                <span className="flex gap-2">
                                    <span className="text-zinc-700 dark:text-zinc-300">12%</span>
                                    <span className="text-green-600 dark:text-green-500">42°C</span>
                                </span>
                            </div>
                            <div className="h-2.5 w-full bg-zinc-100 dark:bg-zinc-800 rounded-full overflow-hidden">
                                <div className="h-full bg-green-500 w-[12%]"></div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* System Load & Charts */}
                <div className="bg-white dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 rounded-xl p-5 flex flex-col gap-4 shadow-sm">
                    <h3 className="text-xs font-bold uppercase tracking-widest flex items-center gap-2 text-zinc-800 dark:text-zinc-200">
                        <Activity size={16} className="text-indigo-500" />
                        LƯU LƯỢNG HỆ THỐNG (SPARK)
                    </h3>

                    <div className="flex-1 flex items-end gap-1.5 h-32 py-2 mt-4">
                        {[40, 60, 45, 90, 65, 30, 85, 20, 50, 70, 95, 40].map((h, i) => (
                            <div
                                key={i}
                                className="flex-1 bg-indigo-500/20 dark:bg-indigo-500/30 hover:bg-indigo-500/50 dark:hover:bg-indigo-500/60 transition-all rounded-t-sm relative group"
                                style={{ height: `${h}%` }}
                            >
                                <span className="absolute -top-7 left-1/2 -translate-x-1/2 text-[10px] font-bold opacity-0 group-hover:opacity-100 transition-opacity bg-zinc-800 text-white dark:bg-white dark:text-zinc-900 px-1.5 py-0.5 rounded z-10 whitespace-nowrap shadow-lg">
                                    {h} MB/s
                                </span>
                            </div>
                        ))}
                    </div>

                    <div className="flex justify-between text-[10px] font-bold text-zinc-400 uppercase tracking-widest mt-1 border-t border-zinc-100 dark:border-zinc-800 pt-2">
                        <span>00:00</span>
                        <span>06:00</span>
                        <span>12:00</span>
                        <span>18:00</span>
                    </div>
                </div>

                {/* Security Metrics */}
                <div className="bg-white dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 rounded-xl p-5 flex flex-col gap-4 shadow-sm">
                    <h3 className="text-xs font-bold uppercase tracking-widest flex items-center gap-2 text-zinc-800 dark:text-zinc-200">
                        <Shield size={16} className="text-sky-500" />
                        BẢO MẬT & PHÂN QUYỀN
                    </h3>
                    <div className="flex flex-col gap-4 flex-1 justify-center">
                        <div className="flex justify-between items-center text-xs">
                            <span className="text-zinc-500 font-medium">Keycloak Status</span>
                            <span className="font-bold text-green-600 dark:text-green-500 bg-green-500/10 px-2 py-1 rounded">Connected</span>
                        </div>
                        <div className="flex justify-between items-center text-xs">
                            <span className="text-zinc-500 font-medium">Active Scholar Tokens</span>
                            <span className="font-bold text-zinc-700 dark:text-zinc-300">12 Session(s)</span>
                        </div>
                        <div className="flex justify-between items-center text-xs">
                            <span className="text-zinc-500 font-medium">Rate Limiting (SlowAPI)</span>
                            <span className="font-bold text-indigo-600 dark:text-indigo-400">0.8req/s</span>
                        </div>
                    </div>
                    <button className="mt-2 w-full py-2 border border-zinc-200 dark:border-zinc-700 text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100 text-[10px] font-bold rounded hover:bg-zinc-50 dark:hover:bg-zinc-800 transition-all uppercase tracking-widest">
                        Quản lý Keycloak Console
                    </button>
                </div>
            </div>

            {/* Live System Logs (Terminal Theme) */}
            <div className="bg-black border border-zinc-800 rounded-xl overflow-hidden flex flex-col shadow-2xl h-80">
                <div className="px-4 py-3 bg-zinc-900/80 border-b border-zinc-800 flex justify-between items-center">
                    <h3 className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest flex items-center gap-2">
                        <Terminal size={14} className="text-zinc-500" />
                        Hán-Nôm System Logs Console
                    </h3>
                    <div className="flex gap-2">
                        <div className="w-2.5 h-2.5 rounded-full bg-red-500 border border-red-600"></div>
                        <div className="w-2.5 h-2.5 rounded-full bg-yellow-500 border border-yellow-600"></div>
                        <div className="w-2.5 h-2.5 rounded-full bg-green-500 border border-green-600"></div>
                    </div>
                </div>

                <div className="p-4 font-mono text-xs leading-relaxed overflow-y-auto space-y-2 flex-1">
                    <p className="text-blue-400"><span className="text-zinc-500 mr-2">[2026-04-07 23:45:01]</span> [INFO] Starting YOLO11n inference on Heritage Scan #4829...</p>
                    <p className="text-zinc-400"><span className="text-zinc-500 mr-2">[2026-04-07 23:45:12]</span> [DEBUG] Identified 142 bounding boxes in segment 01.</p>
                    <p className="text-blue-400"><span className="text-zinc-500 mr-2">[2026-04-07 23:45:14]</span> [INFO] Triggering Qwen2.5-VL for Hán-Nôm character recognition.</p>
                    <p className="text-yellow-400"><span className="text-zinc-500 mr-2 text-zinc-500">[2026-04-07 23:45:20]</span> [WARN] Low contrast detected on Region #12, applying CLAHE enhancement.</p>
                    <p className="text-blue-400"><span className="text-zinc-500 mr-2">[2026-04-07 23:45:22]</span> [INFO] Uploading digitization results to Delta Lake cluster.</p>
                    <p className="text-red-500 font-medium bg-red-500/10 inline-block px-1 rounded"><span className="text-red-400/50 mr-2">[2026-04-07 23:45:30]</span> [ERROR] Redis cache miss for token '南', fetching from PostgreSQL.</p>
                    <p className="text-blue-400"><span className="text-zinc-500 mr-2">[2026-04-07 23:46:01]</span> [INFO] Worker 04 completed task. Waiting for next in queue...</p>

                    <div className="flex items-center mt-4">
                        <span className="text-zinc-500 mr-2">root@hannom-server:~#</span>
                        <div className="w-2 h-4 bg-zinc-300 animate-pulse ml-1"></div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Admin;
