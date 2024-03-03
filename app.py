import modal
from PIL import Image

stub = modal.Stub("table-transformer")

image = (
    modal.Image.micromamba(python_version="3.10")
    .pip_install(
        [
            "scikit-image==0.20.0",
            "scipy==1.10.1",
            "PyMuPDF==1.21.1",
        ]
    )
    .micromamba_install(
        [
            "pytorch==1.13.1",
            "cudatoolkit==11.8.0",
            "torchvision==0.14.1",
            "pandas==1.5.3",
            "scikit-learn==1.2.1",
            "tqdm==4.65.0",
            "Cython==0.29.33",
            "matplotlib==3.7.0",
            "numpy==1.24.2",
            "pycocotools==2.0.6",
            "Pillow==9.4.0",
            "editdistance=0.6.2",
        ],
        channels=["conda-forge", "pytorch", "defaults"],
    )
)

with image.imports():
    from inference import TableExtractionPipeline


@stub.function(image=image)
def predict(img: Image.Image, tokens):
    pipe = TableExtractionPipeline(
        det_config_path="detection_config.json",
        det_model_path="../pubtables1m_detection_detr_r18.pth",
        det_device="cuda",
        str_config_path="structure_config.json",
        str_model_path="../pubtables1m_structure_detr_r18.pth",
        str_device="cuda",
    )
    extracted_tables = pipe.recognize(img, tokens, out_objects=True, out_html=True)
    extracted_table = extracted_tables[0]
    html = extracted_table["html"]
    return html


@stub.local_entrypoint()
def main():
    image = Image.open("table-test.png")
    predict.remote(image, [])
