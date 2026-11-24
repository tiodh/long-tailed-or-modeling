# Symptom → Cure: kegagalan LLM optimization modelling (SFT Qwen2.5-7B-Instruct)

Tanggal: 2026-09-13 · Skrip: `scripts/ast_error_taxonomy.py`, `scripts/wrong_value_symptoms.py`
Data: `results/ast_taxonomy/*` · Catatan: CHANGELOG 2026-09-13 (malam)

## Metode
- Tiap program yang dihasilkan: ekstrak kode → `ast.parse` → fitur statis AST → eksekusi ulang
  dengan stderr ditangkap (SCIPSandbox asli membuang stderr) → kelompok kegagalan.
- Sahih terhadap evaluasi resmi: in-dist 838 benar / 236 wrong_value (resmi 837/242);
  6 benchmark 1084 / 598 (identik).
- Symptom wrong_value: struktur generated vs **reference** (in-dist: `data/ref_structures.jsonl`;
  LogiOR: reference gurobipy ditangkap lewat mock). Multiset koefisien invariant nama & skala.
  Soal yang BENAR dipakai sebagai kontrol (detektor harus jarang berbunyi).
- 6 benchmark tidak punya reference → wrong_value di sana tidak bisa dibedah.

## A. Komposisi kegagalan (% dari semua kegagalan SFT)

| kelompok | in-dist (277/1115) | 6 benchmark (939/2023) | LogiOR (83/92) |
|---|---|---|---|
| format / tanpa kode | 0,0 | 2,6 | 4,8 |
| syntax error | 1,1 | 0,1 | 1,2 |
| konstruksi non-MILP | 2,9 | 9,7 | 20,5 |
| identifier tak terdefinisi (NameError) | 0,4 | 5,9 | 4,8 |
| salah pakai API | 0,0 | 2,0 | 7,2 |
| timeout | 2,5 | 0,5 | 1,2 |
| infeasible / unbounded | 7,9 | 15,5 | 19,3 |
| wrong_value (jalan, optimum salah) | 85,2 | 63,7 | 41,0 |
| **lapis: gagal bangun / gagal solve / tersembunyi** | 6,9 / 7,9 / 85,2 | 20,8 / 15,5 / 63,7 | 39,8 / 19,3 / 41,0 |

Konstruksi non-MILP (dari pesan error): strict inequality `<`/`>` ("can only support with '<=', '>=', or '=='",
60/91 di 6 bench); `bool` Python ke `addCons` ("not ExprCons but bool"); `//`, `%`, `int()`, `range(var)`
pada variabel; perbandingan berantai `lhs <= (expr <= rhs)`.

Fitur AST penanda risiko: variabel biner pada wrong_value vs benar = 24,2 vs 8,1% (in-dist), 14,0 vs 4,5%
(6 bench), 32,4 vs 0,0% (LogiOR). `max/min/abs` pada variabel 0,4 → 2,0 → 4,6%. Constraint di dalam loop
0,0 → 1,5 → 12,6% (gaya in-dist 100% unrolled). Median `addCons` 4 / 4 / 8.

## B. Symptom wrong_value vs reference (% kasus; kontrol = soal benar)

| symptom | in-dist wrong (236) | kontrol benar (838) | LogiOR wrong (33) | kontrol (8) |
|---|---|---|---|---|
| O-coef — isi/koefisien objective salah | **70,3** | 7,4 | **75,8** | 37,5 |
| C-topology — bentuk constraint salah | 27,1 | 10,1 | **81,8** | 25,0 |
| C-missing — constraint kurang | 16,9 | 9,3 | **75,8** | 12,5 |
| C-extra | 15,7 | 8,0 | 12,1 | 12,5 |
| V-missing — variabel kurang | 11,0 | 3,8 | **60,6** | 12,5 |
| V-extra | 12,7 | 3,3 | 3,0 | 0,0 |
| V-int-relaxed — integer ditulis kontinu | 8,1 | 2,0 | **45,5** | 12,5 |
| V-int-over | 9,7 | 4,7 | 12,1 | 25,0 |
| C-rhs | 7,6 | 1,6 | 42,4 | 0,0 |
| C-coef | 6,8 | 0,7 | 27,3 | 12,5 |
| O-sense — arah min/max | 0,0 | 0,0 | 0,0 | 0,0 |

In-dist: kombinasi tersering = "O-coef saja" (90/236). 15/236 tanpa symptom terdeteksi.
Contoh O-coef terverifikasi manual: tarif bertingkat dihitung salah (40/53,75 vs 10/15/20); suku kuadrat
dibuang (objective nonlinear dilinierkan diam-diam); diskon diterapkan keliru (−11 vs −10).

**Koreksi narasi lama "objective 0%":** R_obj lama hanya mengecek ARAH. Gaya ReSocratic menulis objective
sebagai constraint `obj == ekspresi`, sehingga salah-isi-objective dulu terhitung sebagai "constraint error".

## C. Symptom → cure

| symptom | cure (semua saat inference / verifikasi, tanpa training ulang) |
|---|---|
| strict inequality | rewrite AST otomatis `<` → `<=` (−1 integer, −ε kontinu) |
| logika Python pada variabel | lint AST + pustaka linearisasi (abs → 2 constraint; max → var bantu; if-then → biner + big-M) |
| NameError | lint tabel simbol sebelum eksekusi → minta perbaikan nama yang hilang |
| salah pakai API | whitelist API pyscipopt + kartu referensi API di prompt |
| infeasible | umpan balik IIS solver (subset constraint yang bertentangan) → loop perbaikan |
| O-coef / C-rhs / C-coef | grounding koefisien: turunkan koefisien lewat kode dari data mentah; cek setiap angka di soal muncul di model; jaga bentuk nonlinear |
| C-topology | template idiom per family (non-overlap, precedence, flow, subtour); bukti: 1-shot non-overlap 1/24 → 8/24 (p=0,023) |
| C-missing / V-missing | checklist cakupan: enumerasi syarat/entitas soal → wajib ada pasangan constraint/variabel |
| V-int-relaxed | lint integralitas: kata kunci "number of / units / trucks" → wajib INTEGER |

Prioritas: **in-dist** → grounding koefisien objective. **OOD logistik** → lint AST (40% kegagalan
adalah gagal bangun) + checklist cakupan (60–76%) + template idiom relasional (topologi 82%).

## Batas
LogiOR kecil (33 wrong / 8 correct) → persentase kasar. Tanpa reference di 6 benchmark.
Arm yang dianalisis: SFT saja. Tidak ada objektif training (GRPO/ORPO/OTPO bersih) yang mengungguli SFT.

## D. Audit bobot token OTPO vs distribusi kesalahan (2026-09-13 malam)

Skrip: `scripts/otpo_weight_audit.py`, `scripts/otpo_weight_vs_symptom.py` (dijalankan di node GPU; omega
dihitung persis seperti `train_otpo.py::omega_ot`). Data: 159 dari 319 pair bersih yang lolos MAXLEN seperti training.

**Implementasi menyimpang dari paper (Li dkk. 2025):** embedding input STATIS (bukan hidden state kontekstual),
Sinkhorn SEIMBANG, bobot = biaya transport per token (bukan marginal rencana unbalanced). Klaim "faithful
arXiv:2505.18720" di CHANGELOG 2026-09-05 tidak akurat.

**Mapping kategori token benar**: bug teoretis `col_offset` (byte) dipakai sbg offset karakter → 0 token salah
label di 319 pair.

**Bobot hampir seragam:** rerata omega prose 0,97 · kode lain 1,00 · objective 1,18 · constraint body 1,17 ·
constraint relation 1,38. 1% bobot tertinggi: 86% prose + kode lain; token teratas = subword deklarasi tipe
("CONT","INU","OUS","INTEGER") dan kata prose (" aims"," wants"). Di dalam kode, baris yang berbeda dgn pasangan
dapat 2x bobot baris yang sama (1,52 vs 0,76) → OT menangkap beda LEKSIKAL, bukan kesalahan semantik.

**Penargetan per symptom rejected (chosen = reference):** O-coef 40,3% pair, rasio 1,30 (4,1% massa);
C-topology 15,7% / 1,52; C-coef 13,2% / 1,53; C-rhs 11,9% / 1,37; C-extra 11,3% / 1,64; C-missing 10,1% / 1,76;
V-* 3,8–4,4% / 1,19–1,76; gagal bangun/solve total 9,4%. → 93–96% gradien jatuh di luar token pembawa kesalahan.
Symptom terbesar (O-coef = angka) paling lemah disasar: token angka mirip di embedding statis → biaya OT rendah.

**Data tidak memuat symptom OOD:** gagal bangun/solve 9,4% di pair vs 36,3% (6 bench) dan 59,1% (LogiOR) dari
kegagalan; konstruksi non-MILP ~0% vs 9,7/20,5%.

**Cure jalur preference:** bobot dari diff AST (baris objective/constraint berbeda + token angka), dan negatif
yang ditargetkan per symptom OOD (strict inequality, constraint relasional dihapus, integer dilemaskan).

## E. Bobot token in-dist vs OOD — TEMUAN PAPER (2026-09-14)

Skrip `scripts/token_weight_indist_ood.py` · hasil `results/token_weight/indist_vs_ood.json`. Semua pasangan dipakai
(tanpa filter MAXLEN): in-dist 319 (ReSocratic-train, 8 sampel/soal), OOD 412 (soal 6 benchmark yang gagal di greedy,
50 sampel/soal; dipakai HANYA untuk analisis). Kegagalan build dilacak ke baris error via traceback.

| error di jawaban salah | %pair in | %pair out | targeting in | targeting out |
|---|---|---|---|---|
| objective coef/terms | 41,4 | 35,0 | 1,27 | 1,07 |
| constraint bentuk salah | 15,7 | 19,7 | 1,67 | 1,10 |
| constraint hilang | 10,7 | 16,0 | 1,80 | 1,23 |
| constraint berlebih | 12,9 | 18,4 | 1,60 | 1,20 |
| koef constraint | 16,0 | 14,1 | 1,44 | 1,20 |
| rhs | 10,3 | 11,7 | 1,36 | 1,15 |
| integer ditulis kontinu | 3,1 | 14,3 | 1,35 | 1,07 |
| variabel hilang | 4,1 | 6,3 | 1,24 | 0,91 |
| variabel berlebih | 5,0 | 8,5 | 1,58 | 0,89 |
| gagal build (baris error) | 5,0 | 14,6 | 1,59 | 1,36 |
| infeasible/unbounded | 9,1 | 14,1 | — | — |

Kode yang berbeda antara jawaban benar & salah: 35,8% → 53,7%. Rerata w baris beda/sama: 1,44/0,78 → 1,15/0,67.
Rerata w objective 1,20 → 0,97; constraint 1,09 → 1,02; prose 0,95 → 1,03.
**Temuan:** bobot berbasis perbedaan menyasar token pembawa kesalahan in-dist (1,3–1,8×) tapi melantur ke seragam di
OOD (0,9–1,4×) karena jawaban benar & salah berbeda di >50% kode; paling lemah justru pada error yang tumbuh di OOD
(integer dilemaskan ×4,6, gagal build ×2,9, constraint hilang ×1,5).

## F. Efektivitas metode + uji M1 (2026-09-14 pagi)

Skorer per-soal format-agnostic sama untuk semua (`scripts/score_val_fa.py`, 6-bench `score_bench_fa.py`), McNemar vs SFT
(`results/effectiveness_table.json`). In-dist = ReSocratic-val 1.115 (reproduksi angka pipeline metric persis).

| metode | in-dist | 6 benchmark | LogiOR |
|---|---|---|---|
| SFT | 75,16 | 53,58 | 9,78 |
| GRPO | +0,54 (p=0,44) | +0,25 (p=0,72) | −2,17 (p=0,63) |
| ORPO-B (1 constraint) | +1,26 (p=0,22) | +0,74 (p=0,44) | — |
| ORPO-C (2 dimensi) | −1,88 (p=0,09) | −2,67 (p=0,006) | — |
| ORPO-A (alami + chosen reference) | −3,68 (p=0,002) | −5,24 (p<0,001) | −1,09 |
| OTPO bersih (alami) | −3,05 (p=0,003) | −0,84 (p=0,35) | −1,09 |

Tak ada yang naik signifikan; 3 turun signifikan. Urutan mengikuti LOKALISASI KONTRAS (makin sempit beda chosen-rejected makin baik).

Uji M1 (`scripts/m1_displacement.py`, `results/m1/displacement.json`): Δlogp SFT→OTPO bersih, teacher-forced.
chosen ALL −0,0016 (prose −0,0031; kode-sama −0,0003; objective +0,0008; constraint +0,0003) · rejected ALL −0,0087
(prose −0,0146; kode-beda −0,0059) · jawaban benar held-out ALL −0,0023 (prose −0,0036; kode −0,0012; objective +0,0011).
→ displacement NYATA tapi di PROSE/penalaran (~−1,7 nats/jawaban ≈ penalaran SFT 5× kurang mungkin), BUKAN di token kode
bersama; token pembawa kesalahan (objective/constraint) nyaris tak berubah (≤0,001).
**Arah:** kontras hanya di kode formulasi (diff AST), lindungi penalaran (mask prose / anchor NLL), negatif minimal-edit
bertarget symptom di kode — "ORPO-B yang dikerjakan benar".

## G. Uji M2 dan M3 (2026-09-14 pagi) — penyebab penurunan

Trainer: `scripts/train_otpo.py` (+arm `6_codediff`, +env `SAVE_EVERY`; backup .bak_*), rangkaian `scripts/run_m2_m3.sh`,
data `results/metrics/{m2_codediff,m3_step*}_val_fa_rows.jsonl`, loss `results/m3/*_train_metrics.jsonl`.
Pasangan (319 bersih), hyperparameter, titik awal identik OTPO bersih. In-dist ReSocratic-val, McNemar vs SFT.

| titik | in-dist | vs SFT | p | loss | pasangan terpisah |
|---|---|---|---|---|---|
| SFT (step 0) | 75,16 | — | — | — | — |
| M3 step 10 | 75,43 | +0,27 | 0,81 | 1,07 | 20% |
| M3 step 20 | 74,35 | −0,81 | 0,37 | 0,53 | 80% |
| M3 step 30 | 74,17 | −0,99 | 0,27 | 0,03 | 100% |
| M3 step 40 | 72,83 | −2,33 | 0,019 | 0,03 | 100% |
| M2 codediff (40 step) | 72,74 | −2,42 | 0,010 | 0,50 (tak jenuh) | — |

- M2 (lokasi sinyal) TIDAK didukung: bobot hanya baris kode beda rusak sama besar dgn OTPO.
- M3 (over-optimasi) didukung SEBAGIAN: kerusakan menumpuk seiring update (step 10 aman, step 40 signifikan), tapi
  kejenuhan loss bukan syarat (M2 rusak tanpa jenuh).
- TIDAK ADA titik manis: tak satu step pun memberi kenaikan → pasangan jawaban alami tak mengandung kontras berguna;
  early stopping hanya mencegah rusak. Retrain OTPO step 40 = 72,83 vs run pertama 72,11 (variasi antar-run ~0,7).
- **Arah:** masalah di DATA (konstruksi pasangan), bukan pembobotan/penjadwalan. Satu-satunya sinyal positif: ORPO-B
  (pasangan sintetis beda tepat satu constraint) → kembangkan pasangan minimal-edit bertarget symptom.
