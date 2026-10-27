# Qualitative Evaluation V2 — V-FiLM Fix Checkpoint
(training job 66667523, step=4500, val_loss=0.3995)

## Key change from V1
V1 (`scripts/qualitative_eval.py`) used three hand-written instruction paraphrases the model never saw during training. This version replaces them with `build_instruction_for_task()`'s actual T1/T2/T3 template logic (`aadp/data/ctclip_feature_dataset.py`) — the exact instruction wording every training run this session (including this checkpoint's, job 66667523) actually samples from at 60/30/10 weights (`scripts/train_ctclip.py:300-314`). T1 is the fixed training string; T2a/T2b use `extract_entities_from_report`/`sentences_containing` on each scan's own ground-truth report (same functions `ctclip_feature_dataset.py` imports); T3a/T3b use the first positive/negative label column from the report CSV's 18 binary abnormality columns (deterministic, unlike training's `random.choice`, for reproducibility).

## Per-scan results

### SCAN: valid_131_a_1

**T1 — Full report**
- Instruction: "Generate a radiology report for this CT scan."
- GENERATED: Trachea and both main bronchi are open. No occlusive pathology was detected in the trachea, right lobar segmental bronchus is collapsed due to secretions observed on follow-up. There are emphysematous changes in both lungs. Millimetric nonspecific nodules were observed in bilateral lung parenchyma. In addition, there are pleuroparenchymal sequelae density increases that cause structural distortion and volume loss in the apex of both lungs upper lobes. The described findings have been evaluated as secondary to previous TB treatment. Pleural effusion-thickening was not detected in both hemithorax. Mediastinal structures cannot be evaluated optimally because contrast material is not given. As far as can be seen; Heart contour and size are normal. Pericardial effusion was not detectable. Atheroma plaques are present in the aorta and coronary arteries. It appears stable. Lymph nodes with short axes less than 1 cm are observed in prevascular, paratracheal, subcarinal areas, mediastinum, which do not reach pathological dimensions according to the previous examination. When examined in the lung paren-chyma window; Nodular lesions measuring approximately
- GROUND TRUTH: Mediastinal main vascular structures and heart examination IV. It could not be evaluated optimally due to lack of contrast. Calibration of vascular structures, heart contour and size are natural. No pericardial, pleural effusion or increased thickness was detected. Trachea, both main bronchi are open and no obstructive pathology is observed. No pathological increase in wall thickness was detected in the thoracic esophagus. No lymph nodes in pathological size and appearance are observed in the mediastinum, in both axillary regions and in the supraclavicular fossa. In the examination made in the lung parenchyma window; Multiple bulla-bleb formations are observed in both lungs, the largest measuring 36 mm in the upper lobe inferior lingular segment on the left and the largest measuring 47x24 mm in the middle lobe medial segment on the right. Multilobar, peripheral subpleural ground-glass density areas are observed in both lungs, and viral pneumonias are considered in the etiology of the findings. The described findings are among the findings frequently encountered in Covid-19 pneumonia. It is recommended to be evaluated together with clinical and laboratory findings. There are smooth interlobular septal thickness increases, which are more prominent in the lower lobes of both lungs. There is a hypodense fluid density lesion measuring 18 mm in diameter, located cortical in the right kidney midzone posterior, as far as can be seen within the borders of unenhanced CT in the upper abdominal sections within the image. It cannot be clearly characterized (cyst?) within the limits of unenhanced CT. Parenchymal calcifications are observed at the level of liver segment 7. No intraabdominal free fluid or loculated collection was detected. No lytic-destructive lesion was observed in the bone structures within the image, and the vertebral corpus heights were preserved. Smooth thin-walled air cysts, which are more clearly observed in the lower lobes of both lungs, and multi-segmental, peripheral subpleural ground-glass density areas in both lungs - areas of density increase compatible with consolidation are observed, and Covid-19 pneumonia is considered in its etiology. Evaluation with clinical and laboratory findings. recommended.

**T2a — Entity: lung**
- Instruction: "Describe the findings related to lung in this CT scan."
- GENERATED: When examined in the lung parenchyma window; Aeration of both lungs is normal and no nodular or infiltrative lesion is detected in the Lung Parenchymal.
- GROUND TRUTH (entity-filtered): In the examination made in the lung parenchyma window; Multiple bulla-bleb formations are observed in both lungs, the largest measuring 36 mm in the upper lobe inferior lingular segment on the left and the largest measuring 47x24 mm in the middle lobe medial segment on the right. Multilobar, peripheral subpleural ground-glass density areas are observed in both lungs, and viral pneumonias are considered in the etiology of the findings. There are smooth interlobular septal thickness increases, which are more prominent in the lower lobes of both lungs. Smooth thin-walled air cysts, which are more clearly observed in the lower lobes of both lungs, and multi-segmental, peripheral subpleural ground-glass density areas in both lungs - areas of density increase compatible with consolidation are observed, and Covid-19 pneumonia is considered in its etiology.

**T2b — Entity: lobe**
- Instruction: "Describe the findings related to lobe in this CT scan."
- GENERATED: When examined in the lung parenchyma window; In both lungs, there are ground-glass densities that tend to merge with septal thickenings and consolidation areas.
- GROUND TRUTH (entity-filtered): In the examination made in the lung parenchyma window; Multiple bulla-bleb formations are observed in both lungs, the largest measuring 36 mm in the upper lobe inferior lingular segment on the left and the largest measuring 47x24 mm in the middle lobe medial segment on the right. There are smooth interlobular septal thickness increases, which are more prominent in the lower lobes of both lungs. Smooth thin-walled air cysts, which are more clearly observed in the lower lobes of both lungs, and multi-segmental, peripheral subpleural ground-glass density areas in both lungs - areas of density increase compatible with consolidation are observed, and Covid-19 pneumonia is considered in its etiology.

**T3a — Positive: consolidation**
- Instruction: "Is there evidence of consolidation in this scan? Answer yes or no."
- GENERATED: No.
- GROUND TRUTH: "Yes."
- Correct: NO

**T3b — Negative: arterial wall calcification**
- Instruction: "Is there evidence of arterial wall calcification in this scan? Answer yes or no."
- GENERATED: No.
- GROUND TRUTH: "No."
- Correct: YES

### SCAN: valid_634_a_1

**T1 — Full report**
- Instruction: "Generate a radiology report for this CT scan."
- GENERATED: Trachea and both main bronchi are open. No occlusive pathology was detected in the trachea, right lobar segmental bronchus is collapsed due to secretions observed on follow-up. There were no mass or infiltrative lesions in both lungs between their ventilated areas. Mediastinal structures cannot be evaluated optimally because contrast material is not given. As far as can be seen; Heart contour size increased (cardiomegaly). The widths of the mediastinum appear natural within normal limits without pericardial effusion. Atheroma plaques are present in the aorta and coronary arteries. It appears that some calcific lymph nodes have developed at this level with millimetric dimensions. Bilateral minimal pleural effusions persisting from previous examination were noted. In addition, there are newly developing bilateral subpleural nodules measuring 5 mm in diameter in the lower lobes of both lung parenchyma. Apart from these findings, no significant difference was found except for an increase in density consistent with linear fibrotic changes accompanied by sequelae change densities in both upper lobe apical segments of both lungs. Not given.
- GROUND TRUTH: No occlusive pathology was detected in the trachea and both main bronchi. Linear density increases, minimal structural distortion and minimal volume loss, which are evaluated in favor of pleuroparenchymal sequelae changes, are observed in both lung apexes. In addition, there is a similar appearance in the laterobasal segment of the lower lobe of the right lung. Occasionally, linear atelectasis is observed in both lungs. In addition, linear density increases are observed in both lungs, especially in the subpelvral areas. There are millimetric nodules in both lungs. When the previous examinations of the patient are examined, it is understood that the many millimetric nodules observed in both lungs have almost completely disappeared. There are minimal emphysematous changes in both lungs. No mass or infiltrative lesion was detected in both lungs. Mediastinal structures cannot be evaluated optimally because contrast material is not given. As far as can be seen; Heart contour and size are normal. The widths of the mediastinal main vascular structures are normal. Millimetric atheroma plaque is observed in the aorta. No pleural or pericardial effusion was detected. There are short lymph nodes less than 1 cm in diameter in the mediastinum and hilar regions. The shortest diameter of the largest of the described lymph nodes was approximately 7 mm. There is no pathological wall thickness increase in the esophagus within the sections. No upper abdominal free fluid-collection was detected in the sections. No pathologically enlarged lymph nodes were observed. There is a hypodense lesion in the left lobe lateral segment of the liver, which cannot be characterized because contrast agent is not given. However, when the patient was evaluated together with his previous examinations, it was understood that he also had previous examinations and that there was no difference in the dimensions. Vertebral corpus heights, alignments and densities within the sections are normal. There are osteophytes in the vertebral corpus corners. Intervertebral disc distances were minimally narrowed. The neural foramina are open. Localized pleuroparenchymal sequelae and atelectasis in both lungs . Emphysematous changes in both lungs

**T2a — Entity: lung**
- Instruction: "Describe the findings related to lung in this CT scan."
- GENERATED: When examined in the lung parenchyma window; Aeration of both lungs is normal and no nodular or infiltrative lesion is detected in the Lung Parenchymal.
- GROUND TRUTH (entity-filtered): Linear density increases, minimal structural distortion and minimal volume loss, which are evaluated in favor of pleuroparenchymal sequelae changes, are observed in both lung apexes. In addition, there is a similar appearance in the laterobasal segment of the lower lobe of the right lung. Occasionally, linear atelectasis is observed in both lungs. In addition, linear density increases are observed in both lungs, especially in the subpelvral areas. There are millimetric nodules in both lungs. When the previous examinations of the patient are examined, it is understood that the many millimetric nodules observed in both lungs have almost completely disappeared. There are minimal emphysematous changes in both lungs. No mass or infiltrative lesion was detected in both lungs. Localized pleuroparenchymal sequelae and atelectasis in both lungs. Emphysematous changes in both lungs.

**T2b — Entity: lobe**
- Instruction: "Describe the findings related to lobe in this CT scan."
- GENERATED: When examined in the lung parenchyma window; In both lungs, there are ground-glass densities that tend to merge with septal thickenings and consolidation areas.
- GROUND TRUTH (entity-filtered): In addition, there is a similar appearance in the laterobasal segment of the lower lobe of the right lung. There is a hypodense lesion in the left lobe lateral segment of the liver, which cannot be characterized because contrast agent is not given.

**T3a — Positive: arterial wall calcification**
- Instruction: "Is there evidence of arterial wall calcification in this scan? Answer yes or no."
- GENERATED: No.
- GROUND TRUTH: "Yes."
- Correct: NO

**T3b — Negative: bronchiectasis**
- Instruction: "Is there evidence of bronchiectasis in this scan? Answer yes or no."
- GENERATED: No.
- GROUND TRUTH: "No."
- Correct: YES

### SCAN: valid_1_a_1

**T1 — Full report**
- Instruction: "Generate a radiology report for this CT scan."
- GENERATED: Trachea and both main bronchi are open. No occlusive pathology was detected in the trachea, right lobar segmental bronchus is collapsed due to secretions observed on non-contrast examination. There are emphysematous changes in both lungs. Millimetric nonspecific nodules were observed in both lung parenchyma. Pleuroparenchymal sequelae density increases were noted at apical levels of both lungs upper lobes. In addition, there are pleural effusion measurements measuring 5 mm thick between bilateral pleura leaves (sequelae?). Mediastinal structures cannot be evaluated optimally because contrast material is not given. As far as can be seen; Heart contour and size are normal. Pericardial effusion-thickening was not observed. Atheroma plaques are present in the aorta and coronary arteries. The widths of the mediastinal main vascular structures appear natural. Thoracic esophagus calibration was normal and no significant tumoral wall thickening was detected. Sliding type hiatal hernia was observed at the lower end of the esophaguses. Lymph nodes with short axes less than 1 cm that did not reach pathological dimensions were observed within the borders of
- GROUND TRUTH: Trachea, both main bronchi are open. Mediastinal main vascular structures, heart contour, size are normal. Thoracic aorta diameter is normal. Pericardial effusion-thickening was not observed. Thoracic esophageal calibration was normal and no significant tumoral wall thickening was detected. No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions were detected. When examined in the lung parenchyma window; A few millimetric nonspecific nodules and mild recessions are observed in the upper lobe and lower lobe of the right lung. Aeration of both lung parenchyma is normal and no infiltrative lesion is detected in the lung parenchyma. Pleural effusion-thickening was not detected. Upper abdominal organs included in the sections are normal. No space-occupying lesion was detected in the liver that entered the cross-sectional area. Bilateral adrenal glands were normal and no space-occupying lesion was detected. Bone structures in the study area are natural. Vertebral corpus heights are preserved. A few millimetric nonspecific nodules and slight recessions in the upper lobe and lower lobe of the right lung.

**T2a — Entity: lung**
- Instruction: "Describe the findings related to lung in this CT scan."
- GENERATED: When examined in the lung parenchyma window; Aeration of both lungs is normal and no nodular or infiltrative lesion is detected in the Lung Parenchymal.
- GROUND TRUTH (entity-filtered): When examined in the lung parenchyma window; A few millimetric nonspecific nodules and mild recessions are observed in the upper lobe and lower lobe of the right lung. Aeration of both lung parenchyma is normal and no infiltrative lesion is detected in the lung parenchyma. A few millimetric nonspecific nodules and slight recessions in the upper lobe and lower lobe of the right lung.

**T2b — Entity: lobe**
- Instruction: "Describe the findings related to lobe in this CT scan."
- GENERATED: There is a 3 mm diameter nodule in the anterior segment of the right lung upper lobe. There are linear atelectasis areas accompanied by pleuroparenchymal sequelae changes and minimal emphysematous appearance, more prominent on the left side in both lungs.
- GROUND TRUTH (entity-filtered): When examined in the lung parenchyma window; A few millimetric nonspecific nodules and mild recessions are observed in the upper lobe and lower lobe of the right lung. A few millimetric nonspecific nodules and slight recessions in the upper lobe and lower lobe of the right lung.

**T3a — Positive: lung nodule**
- Instruction: "Is there evidence of lung nodule in this scan? Answer yes or no."
- GENERATED: Yes.
- GROUND TRUTH: "Yes."
- Correct: YES

**T3b — Negative: arterial wall calcification**
- Instruction: "Is there evidence of arterial wall calcification in this scan? Answer yes or no."
- GENERATED: No.
- GROUND TRUTH: "No."
- Correct: YES

### SCAN: valid_382_c_2

**T1 — Full report**
- Instruction: "Generate a radiology report for this CT scan."
- GENERATED: Trachea, both main bronchi are open. Mediastinal main vascular structures, heart contour, size are normal. Thoracic aorta diameter is normal. Pericardial effusion-thickening was not observed. Calcific plaques were present in the coronary arteries and LAD (coronary artery disease?). There are calcifications on the walls of the ascending aortic arches. The cardiothoracic index increased in favor of the heart. Pleural effusions reaching 1 cm thickness between bilateral pleura and atelectasis adjacent to it have just appeared. In addition, there are newly developed peribronchovascular thickenings that tend to merge with each other along the lower lobe bronchioles bilaterally. Bilateral lung parenchymal findings consistent with viral pneumonia showed regression according to previous examination. No significant difference was found in nodules evaluated as subpleural nonspecific millimetric pulmonary nodule in right upper lobectomy localization. Other findings described above are stable. Not given.
- GROUND TRUTH: Posterior nodular infiltrates in the right upper lobe decreased in size. The largest is regressed from 25 mm to 19 mm. No newly developed focus of infiltration was observed. Mediastinal lymph nodes are stable. Not given.

**T2a — Entity: lobe**
- Instruction: "Describe the findings related to lobe in this CT scan."
- GENERATED: There are linear atelectasis in the right lung middle lobe medial segment and left lung upper lobe lingular segments.
- GROUND TRUTH (entity-filtered): Posterior nodular infiltrates in the right upper lobe decreased in size.

**T2b — Entity: lymph node**
- Instruction: "Describe the findings related to lymph node in this CT scan."
- GENERATED: No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions were detected.
- GROUND TRUTH (entity-filtered): Mediastinal lymph nodes are stable.

**T3a — Positive: lung nodule**
- Instruction: "Is there evidence of lung nodule in this scan? Answer yes or no."
- GENERATED: Yes.
- GROUND TRUTH: "Yes."
- Correct: YES

**T3b — Negative: arterial wall calcification**
- Instruction: "Is there evidence of arterial wall calcification in this scan? Answer yes or no."
- GENERATED: No.
- GROUND TRUTH: "No."
- Correct: YES

### SCAN: valid_500_d_1

**T1 — Full report**
- Instruction: "Generate a radiology report for this CT scan."
- GENERATED: Trachea and both main bronchi are open. No occlusive pathology was detected in the trachea, right lobar segmental bronchus of a patient with primary lung Ca followed up from our clinic. Mediastinal structures cannot be evaluated optimally because contrast material is not given. As far as can be observed: Heart contour and size are normal. There is minimal pleural effusion on the left. Minimal pericardial thickening was also noted. The widths of the mediastinal main vascular structures are normal except for calcific plaques that cause stenosis at their levels. Lymph nodes were observed in prevascular localization in pathological dimensions measuring 11 mm (8x7mm) short axis). In addition, there are lymphadenopathies located bilaterally in upper-lower paratracheal subcarinal precarinal areas, some of which have lost their fusiform configuration due to postoperative changes. When examined in the lung parenchyma window; Consolidation area including airbronchograms extending towards the lower lobes of both lungs and accompanied by ground glass densities around it has been primarily considered secondary to pneumonic infiltration. It may belong to infective processes or sequelae change-atelectasis. Clinical
- GROUND TRUTH: Mediastinal structures were evaluated as suboptimal since the examination was unenhanced; as far as can be traced; An increase in glandular tissue compatible with gynecomastia was observed in the bilateral retroareolar area. No occlusive pathology was detected in the trachea and left main bronchus lumen. Heart size has increased (cardiomegaly). Pericardial effusion-thickening was not observed. The ascending aorta was 40mm, the pulmonary artery diameter was 30mm, the right pulmonary artery diameter was 28mm, and the left pulmonary artery diameter was 27mm and increased. Diffuse calcified atherosclerotic plaques were observed on the thoracic aorta and coronary artery walls, and densities of stent materials were observed on the coronary artery wall. Thoracic esophagus calibration was normal and no significant pathological wall thickening was detected in the examination borders. When both lung parenchyma windows are evaluated; Emphysematous changes were observed in both lungs. Pleuroparenchymal sequelae density increases were observed in the anterior segment of the right lung upper lobe. Fibtoatelectatic changes were observed in the lateral segment of the middle lobe of the right lung and the inferior lingular segment of the left lung. Soft tissue density, which obliterates the upper lobe bronchus and protrudes in the lumen of the main bronchus, which contains calcification, was observed in the right hilar region. However, in the lesion described distal, large areas of atelectasis-consolidation with indistinguishable borders and increases in ground glass density were observed in its vicinity. The described area of atelectasis-consolidation almost completely fills the upper lobe. It just appeared in the current review. Prominent interlobular septa were observed in the lower lobes of both lungs (secondary to cardiac pathology?). No pleural effusion was detected on the left. Upper abdominal organs included in the sections are normal. No space-occupying lesion was detected in the liver that entered the cross-sectional area. Bilateral adrenal glands were normal and no space-occupying lesion was detected. Degenerative changes were observed in the bone structures in the study area. There is rotoscoliosis with the opening facing left. Mediastinal stable lymph nodes.  Cardiomegaly. Fusiform dilatation of the ascending aorta, dilatation of the pulmonary arteries. Fibroatelectatic changes in both lungs.

**T2a — Entity: lung**
- Instruction: "Describe the findings related to lung in this CT scan."
- GENERATED: When examined in the lung parenchyma window; Aeration of both lungs is normal and no nodular or infiltrative lesion is detected in the Lung Parenchymal.
- GROUND TRUTH (entity-filtered): When both lung parenchyma windows are evaluated; Emphysematous changes were observed in both lungs. Pleuroparenchymal sequelae density increases were observed in the anterior segment of the right lung upper lobe. Fibtoatelectatic changes were observed in the lateral segment of the middle lobe of the right lung and the inferior lingular segment of the left lung. Prominent interlobular septa were observed in the lower lobes of both lungs (secondary to cardiac pathology?). Fibroatelectatic changes in both lungs.

**T2b — Entity: lobe**
- Instruction: "Describe the findings related to lobe in this CT scan."
- GENERATED: There are linear atelectasis in the middle lobe of the right lung and both lungs.
- GROUND TRUTH (entity-filtered): Pleuroparenchymal sequelae density increases were observed in the anterior segment of the right lung upper lobe. Fibtoatelectatic changes were observed in the lateral segment of the middle lobe of the right lung and the inferior lingular segment of the left lung. Soft tissue density, which obliterates the upper lobe bronchus and protrudes in the lumen of the main bronchus, which contains calcification, was observed in the right hilar region. The described area of atelectasis-consolidation almost completely fills the upper lobe. Prominent interlobular septa were observed in the lower lobes of both lungs (secondary to cardiac pathology?).

**T3a — Positive: arterial wall calcification**
- Instruction: "Is there evidence of arterial wall calcification in this scan? Answer yes or no."
- GENERATED: No.
- GROUND TRUTH: "Yes."
- Correct: NO

**T3b — Negative: bronchiectasis**
- Instruction: "Is there evidence of bronchiectasis in this scan? Answer yes or no."
- GENERATED: No.
- GROUND TRUTH: "No."
- Correct: YES

## Cross-scan T2 consistency (entity="lung", all 5 scans)

| Scan | Generated output |
|---|---|
| valid_131_a_1 | When examined in the lung parenchyma window; Aeration of both lungs is normal and no nodular or infiltrative lesion is detected in the Lung Parenchymal. |
| valid_634_a_1 | When examined in the lung parenchyma window; Aeration of both lungs is normal and no nodular or infiltrative lesion is detected in the Lung Parenchymal. |
| valid_1_a_1 | When examined in the lung parenchyma window; Aeration of both lungs is normal and no nodular or infiltrative lesion is detected in the Lung Parenchymal. |
| valid_382_c_2 | When examined in the lung parenchyma window; Aeration of both lungs parenchymes is normal and no nodular or infiltrative lesion is detected in the Lung parenchyme. |
| valid_500_d_1 | When examined in the lung parenchyma window; Aeration of both lungs is normal and no nodular or infiltrative lesion is detected in the Lung Parenchymal. |

Mean pairwise cosine: 0.98
Verdict: [STILL COLLAPSING]

## Summary table

| Task | Metric | Result |
|---|---|---|
| T1 | Distinct outputs across scans | 5/5 |
| T2 | Cross-scan distinct (lung entity) | 1/5 |
| T3 | Binary accuracy | 7/10 |

