# 12. 新しい品番の追加

品番の追加は**共有設定**の変更であり、全設備に配布されます。
設備固有の調整値（オフセット・クロップ）は、追加後に
[品番別調整](fine_adjustment.md#品番別ウィジェット別調整) で行ってください。

## 追加に必要なもの

| 種類 | 場所 | 例 |
|------|------|----|
| 検査 UI | `aikensa/gui/pages/<メーカー>/<シリーズ>/<品番>.ui` | `P658107YA0A.ui` |
| 検査ロジック | `aikensa/core/config/parts/<メーカー>/<シリーズ>/<品番>.py` | `P658107Y0A_SEALASSYRADCORE.py` |
| 仕様（ピッチ・公差） | `aikensa/core/config/parts_specifications.yaml` | `P658107YA0A:` セクション |
| レジストリ登録 | `aikensa/core/config/part_registry.yaml` | `parts:` に 1 エントリ |
| AI モデル | `aikensa/core/config/models.yaml` + `.pt` ファイル | `checkpoints/...` |
| ページ登録 | `aikensa/AIKensa.py` の `UI_FILES` | 末尾に追加 |

## 手順

1. **UI ファイルを作成**
   Qt Designer で既存品番の `.ui` を複製して編集します（[qt_designer.md](qt_designer.md)）。
   - `objectName` は既存の命名規則（`P1label`, `P2label`, …）を守ってください。
   - 既存ページの index は変更しないでください。

2. **検査ロジックを作成**
   既存の品番モジュールを雛形にします。シグネチャは必ず同じにしてください。

   ```python
   def partcheck(image, sahi_predictionList, leftKeypoint, rightKeypoint, expected_side=None):
       ...
       return image, measuredPitch, resultPitch, resultid, status, ng_reason
   ```

3. **仕様を登録**
   `parts_specifications.yaml` に以下を追加します。

   ```yaml
   parts:
     P658107YA0A:
       name: "Seal Assy Rad Core"
       category: "NISSAN"
       series: "M_J42U"
       pitchSpec: [...]
       tolerance_pitch: [...]
       idSpec: [...]
       pixelMultiplier: 0.1655   # mm/pixel
   ```

   `pixelMultiplier` は **ピクセルあたりのミリメートル**です（1 px = 何 mm）。

4. **レジストリに登録**
   `part_registry.yaml` の `parts:` に追記します。

   ```yaml
   - widget_id: 50
     part_number: "P658107YA0A"
     part_name: "Seal Assy Rad Core"
     module_path: "aikensa.core.config.parts.NISSAN.M_J42U.P658107Y0A_SEALASSYRADCORE"
     partcheck_function: "partcheck"
     signal_name: "P658107YA0A_InspectionResult_PitchMeasured"
     directory_name: "P658107YA0A"
     inspection_type: "keypoint"
   ```

5. **`UI_FILES` に追加**
   `aikensa/AIKensa.py` の `UI_FILES` の**末尾**に追記します
   （既存 index をずらさないため）。

6. **動作確認**
   - 自動テスト: `tools\run_tests.cmd`
   - 実機: 基準 OK サンプルと NG サンプルで判定を確認する
   - 結果が `inspection_results` に保存されることを確認する

## 注意

- 既存の `widget_id` を再利用しないでください（履歴・カウンタが混ざります）。
- `part_registry.yaml` に未登録の品番は、レガシー分岐で処理されます。
  新規追加では必ず登録してください。
- 公差（`tolerance_pitch`）の変更は OK/NG 判定に直結します。
  基準サンプルでの再検証なしに変更しないでください。
