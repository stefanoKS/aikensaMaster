# 実機検証チェックリスト（ハードウェア必須）

このチェックリストは、ソフトウェア自動テストでは検証できない項目です。
**実機（IC4 カメラ・校正ボード・PLC）がある環境でのみ実施してください。**

自動テストの結果をもって「実機で動作確認済み」と表現してはいけません。

## 判定の凡例

- **PASS** … 実機で確認し、期待通り
- **FAIL** … 実機で確認し、期待と異なる
- **BLOCKED** … 実機/承認が無く実施できない
- **NOT TESTED** … 未実施（このリポジトリの現状）

> **本リポジトリの実装時点では、下記はすべて NOT TESTED です。**
> 開発環境に IC4 カメラ・校正ボード・PLC が接続されていないためです。

## 1. カメラ接続

- [ ] NOT TESTED — 設備設定 → カメラ設定 → 「カメラ検索 / 更新」で実シリアルが一覧に出る
- [ ] NOT TESTED — カメラ① に正しいシリアルを設定し「接続テスト」で「接続OK」になる
- [ ] NOT TESTED — カメラ② も同様
- [ ] NOT TESTED — 実測値欄に露光・ゲイン・WB の読み戻し値が表示される
- [ ] NOT TESTED — 2 台同時に開いても取り違えが無い（左右が正しい）
- [ ] NOT TESTED — 「カメラ未接続」プレースホルダが生産画像として扱われない

## 2. 内部校正

- [ ] NOT TESTED — 既存設備のカメラ①・②の内部校正とホモグラフィが、上書きせずに設備設定で有効と確認できる
- [ ] NOT TESTED — ChArUco ボードが安定して検出される（照明・ピント）
- [ ] NOT TESTED — 10 フレーム以上で校正が成功する
- [ ] NOT TESTED — 再投影誤差が実運用上許容できる値になる
- [ ] NOT TESTED — `Calibration_camera_0.yaml` / `Calibration_camera_1.yaml` が更新される
- [ ] NOT TESTED — 故意に失敗させた場合、既存ファイルが保持される
- [ ] NOT TESTED — 劣化画像（レンズ歪み）が補正される

## 3. ホモグラフィ / 結合

- [ ] NOT TESTED — カメラマージ画面で両カメラのボードが検出される
- [ ] NOT TESTED — ホモグラフィ計算後に結合画像がテンプレートに重なる
- [ ] NOT TESTED — 結合画像の継ぎ目に不自然な欠けが無い（マスク修正の確認）
- [ ] NOT TESTED — `homography_param_cam1/2.yaml` が更新される
- [ ] NOT TESTED — 平面化変換 (`planarizeTransform_wide.yaml`) が保存される

## 4. 全体微調整

- [ ] NOT TESTED — X/Y を数 px 動かすと結合画像が実際にずれる
- [ ] NOT TESTED — 回転を入れると画像中心を軸に回転する
- [ ] NOT TESTED — 保存前は検査に反映されない（ドラフト）
- [ ] NOT TESTED — 保存後、稼働中の検査に反映される
- [ ] NOT TESTED — 版のロールバックで元の位置に戻る

## 5. 品番別調整

- [ ] NOT TESTED — 品番ごとのオフセットがその品番だけに効く
- [ ] NOT TESTED — 切り出し範囲が実際の検査クロップと一致する
- [ ] NOT TESTED — 「他品番からコピー」が期待通り
- [ ] NOT TESTED — 「共通既定値に戻す」で共有既定値に戻る

## 6. 検査判定（最重要）

- [ ] NOT TESTED — 基準 OK サンプルが OK になる
- [ ] NOT TESTED — 基準 NG サンプルが NG になる（NG 理由も正しい）
- [ ] NOT TESTED — P828387YA1A のマーカーID検査が厳密である
      （`AIKENSA_DEBUG_ACCEPT_ANY_MARKING` 未設定で、別品番を誤って OK にしない）
- [ ] NOT TESTED — 校正欠落時に OK にならない
- [ ] NOT TESTED — モデル欠落時に OK にならない
- [ ] NOT TESTED — 検査時間がリファクタリング前と同等（体感・ログで比較）

## 7. カウント / 梱包 / ピッキング

- [ ] NOT TESTED — OK/NG カウントが正しく増える
- [ ] NOT TESTED — 日付切替で当日カウントがリセットされる
- [ ] NOT TESTED — クリップ投入順 (`pickingOrderSignal`) が従来通り
- [ ] NOT TESTED — 音（OK/NG/計測/梱包）が鳴る

## 8. PLC (SiO)

- [ ] NOT TESTED — サーバ状態表示が ON になる
- [ ] NOT TESTED — 入力信号（検査トリガ）で検査が開始される
- [ ] NOT TESTED — 結果が PLC へ正しく出力される

## 9. データベース

- [ ] NOT TESTED — SQLite (`inspection_results/database_results.db`) に 1 件記録される
- [ ] NOT TESTED — MySQL 接続時に同じ内容が記録される
- [ ] NOT TESTED — レコードのスキーマ（列名・順序）が従来と同一

## 10. 起動 / 終了

- [ ] NOT TESTED — 実機で起動し、メイン画面が全画面表示される
- [ ] NOT TESTED — 「設備設定」ボタンで設定画面が開き、「メイン画面へ戻る」で戻れる
- [ ] NOT TESTED — 終了時にスレッドが停止し、プロセスが残らない
- [ ] NOT TESTED — 校正→検査→校正 の切替を 10 回繰り返しても異常が無い

## 記録欄

| 項目 | 日付 | 実施者 | 判定 | 備考 |
|------|------|--------|------|------|
|      |      |        |      |      |
