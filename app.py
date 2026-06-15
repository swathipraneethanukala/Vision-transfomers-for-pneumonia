import io
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Initialize Flask application
app = Flask(__name__)
CORS(app)  # Enable cross-origin calls from your HTML frontend

# =====================================================================
# 1. YOUR EXACT CUSTOM VISION TRANSFORMER ARCHITECTURE FROM THE NOTEBOOK
# =====================================================================

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)  # (batch_size, embed_dim, h_patches, w_patches)
        x = x.flatten(2).transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn_weights = attn_scores.softmax(dim=-1)
        
        out = (attn_weights @ v).transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        return self.o_proj(out)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, mlp_ratio=4.0):
        super().__init__()
        self.layernorm_before = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadAttention(embed_dim, n_heads)
        self.layernorm_after = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim)

    def forward(self, x):
        x = x + self.attention(self.layernorm_before(x))
        x = x + self.mlp(self.layernorm_after(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, n_classes=2, embed_dim=768, depth=12, n_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))
        
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio) for _ in range(depth)
        ])
        
        self.layernorm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.layernorm(x)
        return self.classifier(x[:, 0])

class PneumoniaModelWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = VisionTransformer()
        
    def forward(self, x):
        return self.vit(x)

# =====================================================================
# 2. MODEL INITIALIZATION & EXACT KEY TRANSLATION MAP
# =====================================================================
print("Instantiating your scratch-built Vision Transformer model...")
model = PneumoniaModelWrapper()

weights_path = 'best_vit_pneumonia_model.pth'

if os.path.exists(weights_path):
    print(f"Loading and remapping custom weights from '{weights_path}'...")
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    
    # Direct Translation Map to solve the exact key mismatch layout
    translated_state_dict = {}
    for key, val in state_dict.items():
        new_key = key
        
        # Translate Embedding components
        if key == "vit.embeddings.cls_token":
            new_key = "vit.cls_token"
        elif key == "vit.embeddings.position_embeddings":
            new_key = "vit.pos_embed"
        elif key == "vit.embeddings.patch_embeddings.projection.weight":
            new_key = "vit.patch_embed.projection.weight"
        elif key == "vit.embeddings.patch_embeddings.projection.bias":
            new_key = "vit.patch_embed.projection.bias"
            
        # Translate Classifier Head components
        elif key == "classifier.weight":
            new_key = "vit.classifier.weight"
        elif key == "classifier.bias":
            new_key = "vit.classifier.bias"
            
        translated_state_dict[new_key] = val
    
    # Load with strict=True to guarantee flawless alignment
    model.load_state_dict(translated_state_dict, strict=True)
    print("✅ Success! Custom weights have successfully bound to your custom architecture layers.")
else:
    print(f"❌ CRITICAL ERROR: '{weights_path}' was not found in the backend folder!")

model.eval()

# 3. EXACT DATA NORMALIZATION FROM YOUR NOTEBOOK CELL
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. ROUTE API TO PROCESS IMAGE REQUESTS
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file segment found'}), 400
        
    file = request.files['file']
    
    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        tensor_img = preprocess(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(tensor_img)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            prediction_idx = torch.argmax(outputs, dim=1).item()
            
        classes = ['NORMAL', 'PNEUMONIA']
        prediction = classes[prediction_idx]
        confidence = probabilities[prediction_idx].item() * 100
        
        print(f"🤖 Diagnostic Process Output: {prediction} | Confidence: {confidence:.2f}%")
        
        return jsonify({
            'prediction': prediction,
            'confidence': confidence
        }), 200

    except Exception as e:
        print(f"Runtime execution error: {str(e)}")
        return jsonify({'error': 'Failed to process matrix context details'}), 500

if __name__ == '__main__':
    print("\n🚀 Starting Scratch-Built ViT Backend Server on http://127.0.0.1:5000")
    app.run(port=5000, debug=True)