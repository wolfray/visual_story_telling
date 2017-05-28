## [The Amazing Mysteries of the Gutter: Drawing Inferences Between Panels in Comic Book Narratives](https://arxiv.org/pdf/1611.05118.pdf)

<p align="justify">
While reading comics, readers rely on commonsense knowledge to connect panels by inferring unseen actions through a process called closure. In this paper, authors examined whether computers can understand the closure-driven narratives conveyed by artwork and dialogue in comic book panels. They presented three novel cloze-style tasks that require a deep understanding of narratives and character to solve.
<p align="justify">

#### Key Points

- They designed three novel tasks, namely, text cloze, visual cloze and character coherence that test a model's ability to understand narratives and characters given a few panels of context.
- Text Cloze: In this task, a model is asked to text from a set of candidates for a particular textbox given context panels (text and image) and the current panel image.
- Visual Cloze: Settings for this task is similar to text cloze but the candidates are images instead of text. A key difference is that models are not given text from final panel, in text cloze, models are allowed to look at the final panel's artwork.
- Character Coherence: This task is associated with character understanding through a re-ordering. Given a jumbled set of text from the textboxes in a particular panel, a model must learn to match each candidate to its corresponding textbox. They restricted this task to panels that contain exactly two dialogue boxes.
- They have set two levels of difficulty (easy and hard) for text cloze and visual cloze.
- They evaluated four different neural models and their best performing model encodes panel with a hierarchical LSTM architecture. They found performance increases when images (VGG-16 features) are given in addition to text. None of the neural architectures outperformed human baselines.
- They presented the COMICS dataset which contains over 1.2 million panels (120 GB) from "Golden Age" comic books.

#### Notes

<p align="justify">
All the baseline architectures differ mainly in the encoding function that converts the sequence of panels into a fixed length vector. To generate image representations, they used fc7 features of VGG-16 and the 4096-d fc7 layer is projected down to the word embedding dimensionality. To train the baseline models, they used cross-entropy loss against the ground-truth labels.
<p align="justify">

**Expermental Dataset**: [Comics Data](https://obj.umiacs.umd.edu/comics/index.html)

**More details on this work**: [Official Implementation and Dataset](https://github.com/miyyer/comics), [MIT Tech Review](https://www.technologyreview.com/s/602973/ai-machine-attempts-to-understand-comic-books-and-fails/)

**Bibliography**
```
@article{iyyer2016amazing,
  title={The Amazing Mysteries of the Gutter: Drawing Inferences Between Panels in Comic Book Narratives},
  author={Iyyer, Mohit and Manjunatha, Varun and Guha, Anupam and Vyas, Yogarshi and Boyd-Graber, Jordan and Daum{\'e} III, Hal and Davis, Larry},
  journal={arXiv preprint arXiv:1611.05118},
  year={2016}
}
```
