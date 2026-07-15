# source_check.md: Meta Kaggle Data source foundation verification (§0.5)

**Verification date**: 2026-07-03 **Conclusion**: ✅ Link link, use Meta Kaggle official dump, do not use the alternative.

harvest.py **is encoded according to the actual listing in this document**, not according to the "expected link" of kb_mining_plan.

## Downloaded files (`kaggle/meta-kaggle`, file-by-file `dataset_download_file`)

| document | size | Loading method |
|---|---|---|
| `Competitions.csv` | 149.6 MB | Can load `usecols` in full |
| `ForumTopics.csv` | 70.4 MB | Can load `usecols` in full |
| `ForumMessages.csv` | **1.70 GB** | **Requires `chunksize` streaming**, filter by Id collection |

The download does not generate `.zip` - `dataset_download_file` of kaggle 2.2.3 directly falls into `.csv`.

## Actual listing (subject to dump this time)

- **Competitions.csv**: `Id, Slug, Title, ForumId, EnabledDate, DeadlineDate, TotalTeams, HostSegmentTitle, Overview, DatasetDescription` (the competition description text is embedded in the last two columns - initial filling of the feature card **no need to crawl the competition webpage**).
- **ForumTopics.csv**:`Id, ForumId, FirstForumMessageId, Title, Score, TotalMessages, CreationDate`.
- **ForumMessages.csv**:`Id, ForumTopicId, PostUserId, PostDate, ReplyToForumMessageId, Message, RawMarkdown, Medal, MedalAwardDate`.

## Confirmation of viable JOIN chain (shorter than planned)

```
Competitions.ForumId  ==  ForumTopics.ForumId          # Press catalog slug Positioning Competition Forum
ForumTopics.FirstForumMessageId  ==  ForumMessages.Id  # Direct foreign key points to the original poster of the original poster!
```

**Key optimization (harvest is implemented based on this)**: `FirstForumMessageId` is a direct foreign key to the first post of the original poster. **There is no need to group and scan ForumMessages** by `ForumTopicId`. Correct approach:

1. Competitions filters out the competitions in catalog → gets the `ForumId` set;
2. ForumTopics filter these ForumId + title regex filter solution posts → get `FirstForumMessageId` collection (together with rank/score/title);
3. ForumMessages **Single chunksize streaming**, `Id ∈ 该集合` takes the text immediately - 1.70 GB only scans once, the hit set is very small (≤10 per competition).

## Text format

- For text, use **`RawMarkdown`** (markdown, LLM is cleaner), and if empty, fall back to `Message` (HTML).
- Note: HTML tags (such as `<b>`) and image links may still be embedded in RawMarkdown, and the truncation/reference verification of extract needs to be tolerated.
- `PostDate` format `MM/DD/YYYY HH:MM:SS` (→ `post_date` of posts.jsonl takes `YYYY-MM-DD`).

## End-to-end demonstration (cassava-leaf-disease-classification)

- ForumId **1000771**, 833 forum posts, the regular title hit **46 posts** solution posts (far more than ≥5).
- 1st place (topic 221957 → FirstForumMessageId **1216990**) Text **8222 characters**, PostDate 02/24/2021, content contains EfficientNet/ResNet/ResNext/ViT/DeiT/MobileNet, "ensemble of four models", best single "B4: 89.4%" - exactly the extract target material.
- False positive observation: The regular rule will mistakenly receive "Can't wait to see 1st team's solution" (wishing post), "Top solutions in..." (summary post) - handed over to the reference verification of extract / "non-scheme post→unclear" filtering, the noise is acceptable.

## Joint correction to catalog (backfilled with true value)

- All 14 races of `start/end` are filled in with the true value of `EnabledDate/DeadlineDate` of dump.
- **`ubc-ocean` slug error → actually `UBC-OCEAN` (all caps)**. The slug filter of harvest must be case-sensitive or aligned with the original value of dump.
- `fathomnet-2025` is only available in **79 teams**, `herbarium-2022-fgvc9` **134 teams**, `sorghum-id-fgvc-9` **252 teams** - write-up may be less than 5, and is determined by the harvest recall policy.

## Alternatives

Not triggered. If structural changes in dump cause chain breakage in the future, see the third-level alternative in kb_mining_plan §0.5.
