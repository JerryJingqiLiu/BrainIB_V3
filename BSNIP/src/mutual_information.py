import torch

class RenyiEntropy:
    """
    A class for computing mutual information using matrix-based entropy calculations.
    This implementation uses Renyi's entropy with matrix-based calculations.
    """
    
    def __init__(self, alpha=1.01):
        """
        Initialize the mutual information calculator.
        
        Args:
            alpha (float): Alpha parameter for Renyi entropy calculation. Default is 5.
        """
        self.alpha = alpha
    
    def _pairwise_distances(self, x):
        """
        Calculate pairwise distances between points in the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (n_samples, n_features)
            
        Returns:
            torch.Tensor: Matrix of pairwise distances
        """
        x = x.view(-1, 1024)  # Flatten to 2D if needed
        instances_norm = torch.sum(x**2, -1).reshape((-1, 1))
        return -2*torch.mm(x, x.t()) + instances_norm + instances_norm.t()
    
    def _calculate_gram_matrix(self, x, sigma):
        """
        Calculate Gram matrix using RBF kernel.
        
        Args:
            x (torch.Tensor): Input tensor
            sigma (float): Bandwidth parameter for RBF kernel
            
        Returns:
            torch.Tensor: Gram matrix
        """
        dist = self._pairwise_distances(x)
        return torch.exp(-dist / sigma)
    
    def renyi_entropy(self, x, sigma):
        """
        Compute Renyi entropy of the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor
            sigma (float): Bandwidth parameter for RBF kernel
            
        Returns:
            torch.Tensor: Renyi entropy value
        """
        k = self._calculate_gram_matrix(x, sigma)
        k = k / torch.trace(k)
        eigv = torch.abs(torch.linalg.eigvalsh(k))
        eig_pow = eigv ** self.alpha
        entropy = (1 / (1 - self.alpha)) * torch.log2(torch.sum(eig_pow))
        return entropy
    
    def joint_entropy(self, x, y, sigma_x, sigma_y):
        """
        Compute joint entropy between two tensors.
        
        Args:
            x (torch.Tensor): First input tensor
            y (torch.Tensor): Second input tensor
            sigma_x (float): Bandwidth parameter for first input
            sigma_y (float): Bandwidth parameter for second input
            
        Returns:
            torch.Tensor: Joint entropy value
        """
        x_gram = self._calculate_gram_matrix(x, sigma_x)
        y_gram = self._calculate_gram_matrix(y, sigma_y)
        k = torch.mul(x_gram, y_gram)
        k = k / torch.trace(k)
        eigv = torch.abs(torch.linalg.eigvalsh(k))
        eig_pow = eigv ** self.alpha
        entropy = (1 / (1 - self.alpha)) * torch.log2(torch.sum(eig_pow))
        return entropy
    
    def calculate_mi(self, x, y, sigma_x, sigma_y):
        """
        Calculate mutual information between two tensors.
        
        Args:
            x (torch.Tensor): First input tensor
            y (torch.Tensor): Second input tensor
            sigma_x (float): Bandwidth parameter for first input
            sigma_y (float): Bandwidth parameter for second input
            
        Returns:
            torch.Tensor: Mutual information value
        """
        h_x = self.renyi_entropy(x, sigma_x)
        h_y = self.renyi_entropy(y, sigma_y)
        h_xy = self.joint_entropy(x, y, sigma_x, sigma_y)
        mi = h_x + h_y - h_xy
        return mi
    
class CS_QMI:
    def __init__(self):
        """
        Initialize the mutual information calculator.
        """
        pass
    
    def _gaussian_matrix(self, X, Y, sigma):
        """
        Calculate Gaussian kernel matrix between two tensors.
        
        Args:
            X (torch.Tensor): First input tensor
            Y (torch.Tensor): Second input tensor
            sigma (float): Bandwidth parameter
            
        Returns:
            torch.Tensor: Gaussian kernel matrix
        """
        size1 = X.size()
        size2 = Y.size()
        G = (X*X).sum(-1)
        H = (Y*Y).sum(-1)
        Q = G.unsqueeze(-1).repeat(1,size2[0])
        R = H.unsqueeze(-1).T.repeat(size1[0],1)
        H = Q + R - 2*X@(Y.T)
        gram_matrix = torch.clamp(torch.exp(-H/2/sigma**2),min=0)
        
        return gram_matrix
    
    def calculate_mi(self, x, y, s_x, s_y):
        """
        Calculate mutual information between two tensors using CS_QMI method.
        
        Args:
            x (torch.Tensor): First input tensor
            y (torch.Tensor): Second input tensor
            s_x (float): Sigma parameter for x kernel
            s_y (float): Sigma parameter for y kernel
            
        Returns:
            torch.Tensor: Mutual information value
        """
        N = x.shape[0]
        
        # Compute two kernel matrices using the provided s_x and s_y
        Kx = self._gaussian_matrix(x, x, sigma=s_x)
        Ky = self._gaussian_matrix(y, y, sigma=s_y)
        
        # Calculate the three terms of CS_QMI
        self_term1 = torch.trace(Kx@Ky.T)/(N**2)
        self_term2 = (torch.sum(Kx)*torch.sum(Ky))/(N**4)
        
        term_a = torch.ones(1,N).to(x.device)
        term_b = torch.ones(N,1).to(x.device)
        cross_term = (term_a@Kx.T@Ky@term_b)/(N**3)
        
        mi = -2*torch.log2(cross_term) + torch.log2(self_term1) + torch.log2(self_term2)
        
        return mi
    
    
