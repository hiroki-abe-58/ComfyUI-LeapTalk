# ComfyUI-LeapTalk（日本語の概要）

**肖像画像と音声から、話す人物の動画を ComfyUI 上でローカル生成します。**

[LeapTalk](https://github.com/zhangrongxiang/LeapTalk)（[論文](https://arxiv.org/abs/2608.00079)、
[重み](https://huggingface.co/z-rx/leaptalk)）を ComfyUI から使うための非公式コミュニティ統合です。
LeapTalk の著者、Soul AI Lab（SoulX-FlashHead）、Meta（wav2vec 2.0）、Comfy Org とは無関係です。

- **公式の重みと推論レシピ**
  - SoulX-FlashHead-1_3B `Model_Pro` に LeapTalk の LoRA と音声投影層を適用し、ViBT scheduler で 1 チャンクあたり 1 ステップで生成します。
  - LeapTalk の公式 `inference.py`（無改変）と同じ画像・音声を与えると、8bit フレームが完全一致しました（234/234 と 850/850 フレーム）。ComfyUI の Load Audio 経由でも同じです。
  - LoRA 480/480 テンソルの適用は、ジョブごとに検証します。
- **入出力**
  - 入力: 画像 1 枚（IMAGE）と音声 1 本（AUDIO）。
  - 出力: 512×512・25fps の VIDEO。元の音声を AAC でそのまま mux し、フレーム数は ceil(音声秒 × 25) です。
  - 音声末尾は切らず、`-shortest` も使いません。
- **実測**（RTX 5090、Windows 11、ネイティブ runtime）
  - 既定の one-shot（ジョブごとに runtime を起動）: キュー投入から完成まで、34 秒の音声で約 45 秒、9.4 秒の音声で約 33〜36 秒、96.5 秒の音声で 71 秒。
  - 1 ジョブにつき約 23 秒は起動とモデル読み込みです（大半が Python の import）。
  - 読み込み後は 28 フレームのチャンクが約 0.45 秒です（生成速度は約 62 フレーム/秒、再生は 25fps）。
- **persistent worker**（v0.2 の新機能、選択式）
  - LeapTalk Runtime ノードの `backend` を `persistent` にすると、最初のジョブで worker を起動してモデルを読み込み、以降の別キューのジョブはそのまま再利用します。
  - 2 件目以降（warm）の実測: 9.4 秒の音声で **6.2 秒**（5 回の中央値。v0.1.0 の one-shot は 36.1 秒）、34 秒の音声で **17.7 秒**（同 45.5 秒）、96.5 秒の音声で 47 秒。
  - 生成フレームは one-shot とも公式スクリプトとも一致しました。worker の最初のジョブは one-shot と同じだけ時間がかかります。
  - **読み込み中の worker は、生成していない間も GPU メモリとシステムメモリを保持します**（本機で CUDA 約 3.5 GiB、Windows のコミット 8.4〜8.8 GiB）。
  - 解放: **LeapTalk Worker** ノードの `unload`、ジョブなしで 120 秒経過（`worker_idle_seconds`）、メモリ逼迫、ComfyUI の終了。`status` は worker を起動せず、待機時間も延長しません。
  - 旧ワークフローと v0.1 の runtime 設定は、変更なしで one-shot のまま動きます。
  - メモリ保護: 「現在のコミット＋推定ピーク＋25% の余裕」が停止水準（95%）に達する見込みなら、モデルの読み込みやジョブの開始を理由と推定不足量付きで拒否し、読み込み中・生成中にコミットが 95% 以上のまま 5 秒続くとジョブと worker を停止します（既定値は本機向けの保守的な設定で、v0.2.1 でも緩めていません）。
  - **v0.2.1 からは one-shot（旧ワークフローを含む）も保護の対象です。** メモリが足りないときは旧ワークフローも拒否されることがあります。別の backend や decoder へ自動で切り替えることはしません。
  - 本機の記録では、ほかのアプリが使うコミットが多いとき Lite TAE の 2 件目以降のジョブは拒否されました（v0.2.0 のクリーン導入では 7 件中 2 件のみ実行）。余裕が少ない環境では `workflows/leaptalk_persistent_lower_memory.json`（persistent＋`wan_vae`。ピークは低いが decode は遅く、Lite TAE とはフレームが異なる）を明示的に選んでください。保護を無効にすることは推奨しません。
- **メモリ**
  - CUDA ピーク: Lite TAE で 8.1/11.1 GiB、Wan VAE で 6.1/7.6 GiB（allocated/reserved）。
  - Windows では CUDA メモリがシステムのコミットにほぼ 1:1 で計上されました。詳細は [docs/BENCHMARKS.md](docs/BENCHMARKS.md) を参照してください。
- **品質の確認範囲**
  - 2 枚の架空の肖像 × 2 つの発話を、単語のタイミングでフレーム確認しました。最初の単語から口が動き、最後の単語も残り、発話終了後に口が閉じます。
  - チャンク境界の差はわずかで、確認したフレーム対に目立つ継ぎ目はありませんでした。
  - 音声を聴いた評価ではありません。
  - audio guidance 2.0 では唇の色や形が崩れたため、既定の 1.0 を推奨します。
- **対応範囲**
  - 検証済み: Windows ネイティブ（ComfyUI とは別の venv）。one-shot と persistent worker の両方。
  - 未対応: WSL2 runtime、macOS（[移植メモ](docs/MACOS.md)）、ライブ配信、複数の ComfyUI プロセスでの worker 共有。
  - 未検証: Linux の実機 GPU、他の GPU、英語以外の音声、96.5 秒を超える音声の実機生成。
- **安全設計**
  - workflow からは実行ファイルもパスも指定できません（管理者が登録した runtime_id を選ぶだけです）。
  - `shell=True` は使わず、環境変数は allowlist で渡します。
  - Windows では Job Object により、キャンセル・タイムアウト・ComfyUI の異常終了のいずれでもプロセスツリーを停止します。persistent worker も同じ Job Object で管理し、ComfyUI を強制終了しても worker の子プロセスは残りませんでした。
  - persistent worker はローカルのパイプで ComfyUI とだけ通信します（ネットワークポートは開きません）。
  - サンドボックスではありません。
- **素材**: デモの肖像（SDXL で生成した架空の人物）と音声（Kokoro-82M）はすべて AI 生成で、権利条件を [docs/LICENSING.md](docs/LICENSING.md) にまとめています。

セットアップ: [docs/SETUP.md](docs/SETUP.md)（英語）

実在の人物の顔や声は、本人の権利と同意がある場合にのみ使い、生成物には AI 生成であることを明記してください。
