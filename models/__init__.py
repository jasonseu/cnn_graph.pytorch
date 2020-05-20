from models.fourier_cgcnn import FourierCGCNN
from models.chebyshev_cgcnn import ChebyshevCGCNN

model_factory = {
    'fourier': FourierCGCNN,
    'chebyshev': ChebyshevCGCNN
}