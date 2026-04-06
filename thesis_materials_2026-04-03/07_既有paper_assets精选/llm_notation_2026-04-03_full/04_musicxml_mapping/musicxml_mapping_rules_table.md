# MusicXML 映射逻辑与规则表

## A. 简谱数字 -> MusicXML `<step>`

| 减字谱推断 pitch | MusicXML step |
|---|---|
| `1` | `C` |
| `2` | `D` |
| `3` | `E` |
| `4` | `F` |
| `5` | `G` |
| `6` | `A` |
| `7` | `B` |

## B. 时值 -> MusicXML `<type>` + `<duration>`

| LLM duration | MusicXML type | MusicXML duration |
|---|---|---|
| `2` | `half` | `32` |
| `4` | `quarter` | `16` |
| `8` | `eighth` | `8` |
| `16` | `16th` | `4` |

## C. 减字四层信息 -> MusicXML `<lyric>`

| lyric number | 语义字段 | 来源字段 |
|---|---|---|
| `1` | 右手技法/动作 | `action` |
| `2` | 弦序 | `string` |
| `3` | 徽位 | `position` |
| `4` | 左手手指/按法 | `finger` |

## D. 结构规则

- 当 `new_measure=true` 时，写入新 `<measure>`。
- `<octave>` 来自 LLM 的 `octave` 字段。
- 生成后进行 lyric 编号完整性检查（1~4 必须齐全）。
