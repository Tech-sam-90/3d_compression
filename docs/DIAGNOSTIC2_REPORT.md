# ICTC Diagnostic Report 2

Read-only diagnostics run against ground-truth CT-RATE reference reports.
No model/training code or checkpoints were modified for anything in this
file.

---

## Collapsed Scan Ground Truth Check

**Goal:** determine whether the 13.3% text-level collapse rate found in
Test 3 Part B (see `docs/DIAGNOSTIC_REPORT.md`, "Test 3 Full Rerun —
checkpoint_best.pt — post 3-epoch training") is a **data property** (the 5
collapsed scans genuinely have similar ground-truth reports) or a **model
failure** (the model is ignoring real clinical differences that exist in
the ground truth).

**Source:** `/project/def-uanazodo-ab/sadeniji/ctrate_csv/dataset/merged/valid_merged.csv`, column `Findings_EN`. `VolumeName` values are `<stem>.nii.gz`; matched by stripping the `.nii.gz` suffix. All 10 scan IDs matched exactly one row each, no ambiguity.

### Step 1-2: Reference reports

**Collapsed cluster** (the 5 scans whose *generated* reports were near-identical in Test 3 Part B — pairs (2,9), (3,6), (3,10), (6,10) exact; (2,7),(2,9),(3,6),(3,10),(6,10),(7,9) near-duplicate):

1. **valid_131_a_1**
   > Mediastinal main vascular structures and heart examination IV. It could not be evaluated optimally due to lack of contrast. Calibration of vascular structures, heart contour and size are natural. No pericardial, pleural effusion or increased thickness was detected. Trachea, both main bronchi are open and no obstructive pathology is observed. No pathological increase in wall thickness was detected in the thoracic esophagus. No lymph nodes in pathological size and appearance are observed in the mediastinum, in both axillary regions and in the supraclavicular fossa. In the examination made in the lung parenchyma window; Multiple bulla-bleb formations are observed in both lungs, the largest measuring 36 mm in the upper lobe inferior lingular segment on the left and the largest measuring 47x24 mm in the middle lobe medial segment on the right. Multilobar, peripheral subpleural ground-glass density areas are observed in both lungs, and viral pneumonias are considered in the etiology of the findings. The described findings are among the findings frequently encountered in Covid-19 pneumonia. It is recommended to be evaluated together with clinical and laboratory findings. There are smooth interlobular septal thickness increases, which are more prominent in the lower lobes of both lungs. There is a hypodense fluid density lesion measuring 18 mm in diameter, located cortical in the right kidney midzone posterior, as far as can be seen within the borders of unenhanced CT in the upper abdominal sections within the image. It cannot be clearly characterized (cyst?) within the limits of unenhanced CT. Parenchymal calcifications are observed at the level of liver segment 7. No intraabdominal free fluid or loculated collection was detected. No lytic-destructive lesion was observed in the bone structures within the image, and the vertebral corpus heights were preserved.

2. **valid_258_a_2**
   > A triangular density is observed secondary to the thymic remnant in the anterior mediastinum. Trachea and main bronchi are open. Right upper-lower paratracheal millimetric size 1-2 lymph nodes are observed. No pathological LAP was detected in the mediastinum. The heart and mediastinal vascular structures have a natural appearance. Pleural effusion-thickening was not detected in both hemithorax. In the evaluation of both lung parenchyma; No mass nodule infiltration was detected in both lungs. In the sections passing through the upper part of the abdomen, the bilateral adrenal glands appear natural. No significant pathology was detected in the abdominal sections. No obvious pathology was detected in bone structures.

3. **valid_634_a_1**
   > No occlusive pathology was detected in the trachea and both main bronchi. Linear density increases, minimal structural distortion and minimal volume loss, which are evaluated in favor of pleuroparenchymal sequelae changes, are observed in both lung apexes. In addition, there is a similar appearance in the laterobasal segment of the lower lobe of the right lung. Occasionally, linear atelectasis is observed in both lungs. In addition, linear density increases are observed in both lungs, especially in the subpelvral areas. There are millimetric nodules in both lungs. When the previous examinations of the patient are examined, it is understood that the many millimetric nodules observed in both lungs have almost completely disappeared. There are minimal emphysematous changes in both lungs. No mass or infiltrative lesion was detected in both lungs. Mediastinal structures cannot be evaluated optimally because contrast material is not given. As far as can be seen; Heart contour and size are normal. The widths of the mediastinal main vascular structures are normal. Millimetric atheroma plaque is observed in the aorta. No pleural or pericardial effusion was detected. There are short lymph nodes less than 1 cm in diameter in the mediastinum and hilar regions. The shortest diameter of the largest of the described lymph nodes was approximately 7 mm. There is no pathological wall thickness increase in the esophagus within the sections. No upper abdominal free fluid-collection was detected in the sections. No pathologically enlarged lymph nodes were observed. There is a hypodense lesion in the left lobe lateral segment of the liver, which cannot be characterized because contrast agent is not given. However, when the patient was evaluated together with his previous examinations, it was understood that he also had previous examinations and that there was no difference in the dimensions. Vertebral corpus heights, alignments and densities within the sections are normal. There are osteophytes in the vertebral corpus corners. Intervertebral disc distances were minimally narrowed. The neural foramina are open.

4. **valid_1016_b_2**
   > Minimal effusion was observed in both pleural spaces. Measured 20 mm on the right at its deepest point. In both lungs, there are areas of increase in density consistent with newly developed consolidation, which is evaluated in favor of compressive atelectasis adjacent to the effusion. In the mediastinum, a lesion of soft tissue density is observed in the prevascular area, which is evaluated primarily in favor of lymphadenopathy, in which calcified foci in millimeter sizes are also observed. Although no change was found in the craniocaudal dimension in the current examination, an increase in the mediolateral dimension was noted. It was measured as 25 mm in the previous CT examination, and it was measured as 31 mm in the current examination. In addition, there are lymph nodes in the mediastinum that are stable in number and size, short in diameter less than 1 cm, have a fusiform configuration, and are not pathological in size and appearance. There are nodules in both lungs, the largest of which is in the posterobasal segment of the left lung lower lobe, some with irregular borders and some with a ground-glass halo in the periphery. No change was detected in their number and size. In addition, thickening in the peribronchovascular area and smooth interlobular septal thickness increases are observed in the anterior segment of the left lung upper lobe. The findings were also observed in the previous CT examination and no change was detected.

5. **valid_1147_b_2**
   > Trachea, both main bronchi are open. Mediastinal main vascular structures, heart contour, size are normal. Thoracic aorta diameter is normal. A smear-like pericardial effusion is observed. Thoracic esophagus calibration was normal and no significant tumoral wall thickening was detected. No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions were detected. A small amount of pleural effusion is observed in the left hemithorax. There is a mosaic attenuation pattern in the basal segment of the lower lobe of the left lung. Upper abdominal organs included in the sections are normal. No space-occupying lesion was detected in the liver that entered the cross-sectional area. Bilateral adrenal glands were normal and no space-occupying lesion was detected. Bone structures in the study area are natural. Vertebral corpus heights are preserved. In the TH5 vertebral corpus, there is a finding consistent with a hemangioma in the first plan, measuring 7 mm in size.

**Distinct group** (5 scans whose generated reports remained meaningfully different from each other and from the collapsed cluster):

6. **valid_1_a_1**
   > Trachea, both main bronchi are open. Mediastinal main vascular structures, heart contour, size are normal. Thoracic aorta diameter is normal. Pericardial effusion-thickening was not observed. Thoracic esophageal calibration was normal and no significant tumoral wall thickening was detected. No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions were detected. When examined in the lung parenchyma window; A few millimetric nonspecific nodules and mild recessions are observed in the upper lobe and lower lobe of the right lung. Aeration of both lung parenchyma is normal and no infiltrative lesion is detected in the lung parenchyma. Pleural effusion-thickening was not detected. Upper abdominal organs included in the sections are normal. No space-occupying lesion was detected in the liver that entered the cross-sectional area. Bilateral adrenal glands were normal and no space-occupying lesion was detected. Bone structures in the study area are natural. Vertebral corpus heights are preserved.

7. **valid_382_c_2**
   > Posterior nodular infiltrates in the right upper lobe decreased in size. The largest is regressed from 25 mm to 19 mm. No newly developed focus of infiltration was observed. Mediastinal lymph nodes are stable.

8. **valid_500_d_1**
   > Mediastinal structures were evaluated as suboptimal since the examination was unenhanced; as far as can be traced; An increase in glandular tissue compatible with gynecomastia was observed in the bilateral retroareolar area. No occlusive pathology was detected in the trachea and left main bronchus lumen. Heart size has increased (cardiomegaly). Pericardial effusion-thickening was not observed. The ascending aorta was 40mm, the pulmonary artery diameter was 30mm, the right pulmonary artery diameter was 28mm, and the left pulmonary artery diameter was 27mm and increased. Diffuse calcified atherosclerotic plaques were observed on the thoracic aorta and coronary artery walls, and densities of stent materials were observed on the coronary artery wall. Thoracic esophagus calibration was normal and no significant pathological wall thickening was detected in the examination borders. When both lung parenchyma windows are evaluated; Emphysematous changes were observed in both lungs. Pleuroparenchymal sequelae density increases were observed in the anterior segment of the right lung upper lobe. Fibtoatelectatic changes were observed in the lateral segment of the middle lobe of the right lung and the inferior lingular segment of the left lung. Soft tissue density, which obliterates the upper lobe bronchus and protrudes in the lumen of the main bronchus, which contains calcification, was observed in the right hilar region. However, in the lesion described distal, large areas of atelectasis-consolidation with indistinguishable borders and increases in ground glass density were observed in its vicinity. The described area of atelectasis-consolidation almost completely fills the upper lobe. It just appeared in the current review. Prominent interlobular septa were observed in the lower lobes of both lungs (secondary to cardiac pathology?). No pleural effusion was detected on the left. Upper abdominal organs included in the sections are normal. No space-occupying lesion was detected in the liver that entered the cross-sectional area. Bilateral adrenal glands were normal and no space-occupying lesion was detected. Degenerative changes were observed in the bone structures in the study area. There is rotoscoliosis with the opening facing left.

9. **valid_890_a_2**
   > Heart contour and size are normal. No pleural-pericardial effusion or thickening was detected. The widths of the mediastinal main vascular structures are normal. A few lymph nodes with a short diameter less than 5 mm are observed in the mediastinum and bilateral hilar regions, and no enlarged lymph nodes in pathological size and appearance are detected. Trachea and both main bronchi are open. No occlusive pathology was detected in the trachea and both main bronchi. Linear atelectasis areas are observed in both lungs. Several nodules with a diameter of 3.5 mm are observed in both lungs, the largest of which is in the lateral segment of the right lung middle lobe. No mass or infiltrative lesion was detected in both lungs. No pathological increase in wall thickness was detected in the esophagus. As far as it can be evaluated within the limits of non-contrast CT; There is no discernible mass in the upper abdominal organs. No lytic-destructive lesions were observed in the bone structures within the sections.

10. **valid_766_a_1**
    > Trachea, both main bronchi are open. Mediastinal main vascular structures, heart contour, size are normal. Pericardial effusion-thickening was not observed. Thoracic esophagus calibration was normal and no significant pathological wall thickening was detected. No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions were detected. When examined in the lung parenchyma window; More than one patchy ground glass densities are observed in both lungs, especially in the lower lobe superior and posterior basal parts. Clinical and laboratory correlation and follow-up are recommended for viral pneumonia. Upper abdominal organs are included in the study partially and no gross pathology was found. A slight decrease in density is observed in the bone structures in the examination area, and there are hypertrophic osteophytic taperings in the end plates of the vertebral corpuscles.

### Step 3: Pairwise edit distance between reference reports

**Collapsed cluster (5 scans, 10 pairs), reference-report edit distances:**

| Pair | Normalized edit distance |
|---|---|
| valid_131_a_1 vs valid_258_a_2 | 0.7516 |
| valid_131_a_1 vs valid_634_a_1 | 0.7239 |
| valid_131_a_1 vs valid_1016_b_2 | 0.6959 |
| valid_131_a_1 vs valid_1147_b_2 | 0.7049 |
| valid_258_a_2 vs valid_634_a_1 | 0.7686 |
| valid_258_a_2 vs valid_1016_b_2 | 0.7276 |
| valid_258_a_2 vs valid_1147_b_2 | 0.7023 |
| valid_634_a_1 vs valid_1016_b_2 | 0.6957 |
| valid_634_a_1 vs valid_1147_b_2 | 0.7206 |
| valid_1016_b_2 vs valid_1147_b_2 | 0.7043 |

**Distinct group (5 scans, 10 pairs), reference-report edit distances:**

| Pair | Normalized edit distance |
|---|---|
| valid_1_a_1 vs valid_382_c_2 | 0.8527 |
| valid_1_a_1 vs valid_500_d_1 | 0.6580 |
| valid_1_a_1 vs valid_890_a_2 | 0.7158 |
| valid_1_a_1 vs valid_766_a_1 | 0.4062 |
| valid_382_c_2 vs valid_500_d_1 | 0.9165 |
| valid_382_c_2 vs valid_890_a_2 | 0.8418 |
| valid_382_c_2 vs valid_766_a_1 | 0.8378 |
| valid_500_d_1 vs valid_890_a_2 | 0.7154 |
| valid_500_d_1 vs valid_766_a_1 | 0.7274 |
| valid_890_a_2 vs valid_766_a_1 | 0.7191 |

**Summary:**

```
Mean edit dist — collapsed cluster refs: 0.7195   (min 0.6957, max 0.7686)
Mean edit dist — distinct group refs:    0.7391   (min 0.4062, max 0.9165)
```

### Step 4: Verdict

**(a) Are the reference reports for the 5 collapsed scans similar to each other (mean edit dist < 0.15)?**
**No.** Mean reference-report edit distance within the collapsed cluster is **0.7195** — roughly 5x above the 0.15 threshold, and clinically the five reports describe unrelated pathology: COVID-pattern pneumonia with bulla-bleb formations and a renal lesion (`valid_131_a_1`), a thymic remnant with borderline lymph nodes and an otherwise normal chest (`valid_258_a_2`), pleuroparenchymal sequelae with hepatic and spinal findings (`valid_634_a_1`), pleural effusion with mediastinal lymphadenopathy and multiple lung nodules (`valid_1016_b_2`), and pericardial effusion with a vertebral hemangioma (`valid_1147_b_2`). **This is a model failure, not a data property** — the ground truth is genuinely diverse, and the model collapsed it into near-identical generated text.

**(b) Are the reference reports for the 5 distinct scans more varied than the collapsed cluster?**
**Only marginally, and not meaningfully.** Distinct-group mean is 0.7391 vs. collapsed-cluster's 0.7195 — both groups' reference reports are comparably diverse (same order of magnitude; the distinct group's wider min/max range of 0.41-0.92 vs. the collapsed cluster's tight 0.70-0.77 is the only real structural difference, and if anything shows the collapsed cluster's references are *slightly more uniformly spread apart*, not more clustered). There is no evidence the "distinct" scans were selected by the model because their ground truth is unusually more varied than the "collapsed" scans' ground truth — both are similarly diverse. This reinforces (a): the model's differentiation (or lack of it) does not track real ground-truth variation for this cluster.

**(c) Word-count / unique-word-count, collapsed cluster reference reports:**

| Scan | Words | Unique words | TTR |
|---|---|---|---|
| valid_131_a_1 | 290 | 159 | 0.5483 |
| valid_258_a_2 | 109 | 69 | 0.6330 |
| valid_634_a_1 | 324 | 156 | 0.4815 |
| valid_1016_b_2 | 240 | 117 | 0.4875 |
| valid_1147_b_2 | 158 | 97 | 0.6139 |

(For contrast, distinct-group reference reports: 156/89/0.5705, 34/31/0.9118, 334/156/0.4671, 171/92/0.5380, 136/98/0.7206 — comparable range.)

The collapsed cluster's reference reports are not short (109-324 words) or unusually repetitive (TTR 0.48-0.63, in line with the distinct group's 0.47-0.91 range, aside from `valid_382_c_2`'s single short 34-word report which trivially inflates its own TTR). **No evidence supports a "short/repetitive reference report" explanation for the collapse either.**

### Overall conclusion

All three checks point the same direction: **the 13.3% Part B collapse rate found in Test 3 is a model failure, not a data property.** The five scans that produced near-identical generated reports have reference reports describing five clinically distinct conditions with a mean pairwise edit distance (0.72) essentially indistinguishable from the "distinct" comparison group (0.74). The aggregator (top-K sparse + cosine visual cross-attention, unchanged from the FiLM pipeline in this attention-conditioned ablation) is failing to encode scan-specific information into the M=64 tokens for this cluster of scans specifically — consistent with the `docs/DIAGNOSTIC_REPORT.md` verdict's hypothesis that the visual cross-attention mechanism, not the conditioning mechanism, is the remaining bottleneck for scan differentiation. Whether this traces back to the underlying CT-CLIP *feature* similarity for these 5 scans specifically (a data-adjacent explanation, distinct from ground-truth-report similarity) versus a genuine aggregator weakness was not tested here and would need a follow-up check of the raw CT-CLIP feature cosine similarity for this specific cluster.
