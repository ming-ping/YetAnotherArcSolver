"""
It's wise to carefully consider the initial image representation in Python as it will significantly impact the subsequent stages of your project. Based on the sources, particularly those addressing the Abstraction and Reasoning Corpus (ARC), we can identify key considerations, tradeoffs, past approaches, and available choices.

**Key Considerations and Tradeoffs:**

*   **Level of Abstraction:**
    *   **Low-level (Pixel-based):** Representing an image as a grid of individual pixel colors (e.g., using integer values for colors) is the most direct approach. This offers fine-grained control but can be computationally intensive for complex operations or large images. The sources heavily utilize this for ARC tasks.
    *   **Higher-level (Object-based):** Identifying and representing distinct "objects" within the image, along with their properties (e.g., shape, size, position, color) and relationships, provides a more abstract view. This can simplify reasoning and certain types of transformations but requires an initial step of object detection/segmentation. The sources show that object-based representations are crucial for solving many ARC tasks as humans tend to think in terms of objects and their relations.
    *   **Tradeoff:** Choosing between these levels involves balancing the need for detailed pixel-level information with the desire for more abstract, computationally efficient representations for reasoning and manipulation. Starting with a low-level representation like a pixel grid allows for the derivation of higher-level features later if needed.

*   **Computational Efficiency:**
    *   Operating directly on pixel grids, especially for larger images, can be computationally demanding.
    *   Higher-level representations that group pixels into objects might lead to more efficient algorithms for certain tasks, as operations can be performed on entire objects rather than individual pixels.
    *   **Tradeoff:** Simpler, direct representations might be easier to implement initially but could lead to performance bottlenecks later, especially if your project involves complex image processing or large datasets.

*   **Flexibility and Expressiveness:**
    *   A pixel-based representation is very general and can represent any type of image.
    *   Object-based representations are powerful for tasks that naturally involve distinct entities but might be less suitable for images without clearly defined objects or for operations that require pixel-level precision.
    *   **Tradeoff:** A more flexible, low-level representation might be a safer starting point if the future requirements of your project are not entirely clear. You can always add layers of abstraction later.

*   **Integration with Libraries:**
    *   Python has excellent libraries for numerical computing (NumPy) and image processing (Pillow, OpenCV, scikit-image).
    *   Representing images as NumPy arrays (multidimensional arrays) is highly advantageous as it directly interfaces with these libraries, enabling efficient operations.
    *   **Consideration:** Choose a representation that can be easily converted to or directly used by these standard libraries.

**Past Approaches (Based on the Sources):**

*   **2D NumPy Arrays:** This is the **most common and fundamental representation** seen in the context of ARC problem solving. The sources describe inputs and outputs of ARC tasks as 2D grids of colored pixels, which are naturally represented by 2D numerical arrays where each cell holds a color ID.
*   **String Encoding:** Some approaches, particularly those involving language models, convert the 2D pixel grid into a 1D string format, with tokens representing pixel colors and delimiters for rows. This allows leveraging the sequence processing capabilities of LMs.
*   **Graph-based Representation:** For more sophisticated reasoning, images are sometimes abstracted into graphs where nodes represent objects (defined by connected pixels of the same color), and edges represent their spatial relationships. This requires a preceding step of identifying objects.
*   **Domain-Specific Language (DSL) Operations:** While not a direct representation, DSLs for ARC define operations that act upon these grid representations. For example, the PeARL language operates on a `Grid` type, which internally likely uses a pixel-based representation.

**Number of Choices Available:**

At a fundamental level for representing a static image in Python, your primary choices revolve around the level of abstraction:

1.  **Direct Pixel Representation:**
    *   **Python Lists of Lists:** A basic way to represent a 2D grid, but less efficient for numerical operations.
    *   **NumPy 2D Array:** The **most highly recommended choice** for image data in Python due to its efficiency and seamless integration with scientific and image processing libraries. You can choose the data type to represent colors (e.g., integers).
    *   **Pillow (PIL) `Image` Object:** This library provides an `Image` class that encapsulates image data and offers many built-in functions for manipulation. You can access and modify pixel data within these objects.

2.  **Abstracted Object Representation:**
    *   **List of Dictionaries/Objects:** Representing the image as a list where each element describes an object (e.g., its pixels, bounding box, color, label).
    *   **Graph Data Structures:** Using libraries like NetworkX to represent objects as nodes and their relationships as edges.

3.  **Hybrid Approaches:**
    *   Combining a core pixel grid (e.g., NumPy array) with auxiliary data structures to store object information or other features.

**Recommendation for Your First Step:**

Given that this is the first and crucial step, and drawing from the prevalence of NumPy arrays in the ARC literature for representing image grids, **starting with a NumPy 2D array to represent your image is likely the most robust and flexible choice**.

*   It provides a direct, low-level representation of the pixel data.
*   It offers excellent performance for numerical operations, which are common in image processing.
*   It seamlessly integrates with a wide range of powerful Python libraries for image manipulation, analysis, and visualization (e.g., Pillow, OpenCV, scikit-image, Matplotlib).

You can always build higher-level abstractions (like object detection and object-based representations) on top of this fundamental pixel grid if your project's needs evolve in that direction. This approach provides a solid foundation without prematurely committing to a specific high-level abstraction that might not be suitable for all future requirements.
"""
import numpy as np
import os, json

class Arcimage:
    @staticmethod
    def random():
        h,w = np.random.choice(30,2)+1
        return Arcimage(np.random.choice(10,(h,w)))
    def __init__(self, data:np.ndarray):
        self.data = data.astype(np.int8)
    def sub_img(self,ir,ic,h,w): return Arcimage(self.data[ir:ir+h,ic:ic+w])
    def random_color(self): return np.random.choice(10)
    def random_shape(self): return np.random.choice(30,2)+1
    def random_image(self,shape): return Arcimage(np.random.choice(10,shape))
    def show(self):
        for row in self.data:
            for v in row:
                if v==0: print("\033[48;5;232m  \033[0m",end="")
                elif v==1: print("\033[48;5;201m  \033[0m",end="")
                elif v==2: print("\033[48;5;124m  \033[0m",end="")
                elif v==3: print("\033[48;5;196m  \033[0m",end="")
                elif v==4: print("\033[48;5;202m  \033[0m",end="")
                elif v==5: print("\033[48;5;220m  \033[0m",end="")
                elif v==6: print("\033[48;5;112m  \033[0m",end="")
                elif v==7: print("\033[48;5;105m  \033[0m",end="")
                elif v==8: print("\033[48;5;117m  \033[0m",end="")
                elif v==9: print("\033[48;5;248m  \033[0m",end="")
                else: print("  ",end="")
            print()
    def __repr__(self):
        return f"Arcimage({self.data})"
    def __str__(self):
        return f"Arcimage({self.data})"
    def __eq__(self, other):
        return np.array_equal(self.data, other.data)
    def __hash__(self):
        return hash(self.data.tobytes())
    # def __len__(self):
    #     return len(self.data)
    # def __getitem__(self, key):
    #     return self.data[key]
    # def __setitem__(self, key, value):
    #     self.data[key] = value
    # def __iter__(self):
    #     return iter(self.data)
    # def __reversed__(self):
    #     return reversed(self.data)
    # def __contains__(self, item):
    #     return item in self.data
    # def __add__(self, other):
    #     return Arcimage(self.data + other.data)
    # def __sub__(self, other):
    #     return Arcimage(self.data - other.data)
    # def __mul__(self, other):
    #     return Arcimage(self.data * other.data)
    # def __truediv__(self, other):
    #     return Arcimage(self.data / other.data)
    # def __floordiv__(self, other):
    #     return Arcimage(self.data // other.data)
    # def __mod__(self, other):
    #     return Arcimage(self.data % other.data)
    # def __pow__(self, other):
    #     return Arcimage(self.data ** other.data)
    # def __lshift__(self, other):
    #     return Arcimage(self.data << other.data)
    # def __rshift__(self, other):
    #     return Arcimage(self.data >> other.data)
    # def __and__(self, other):
    #     return Arcimage(self.data & other.data)
    # def __xor__(self, other):
    #     return Arcimage(self.data ^ other.data)
    # def __or__(self, other):
    #     return Arcimage(self.data | other.data)
    # def __radd__(self, other):
    #     return Arcimage(self.data + other.data)
    # def __rsub__(self, other):
    #     return Arcimage(self.data - other.data)
    # def __rmul__(self, other):
    #     return Arcimage(self.data * other.data)
    # def __rtruediv__(self, other):
    #     return Arcimage(self.data / other.data)
    # def __rfloordiv__(self, other):
    #     return Arcimage(self.data // other.data)
    # def __rmod__(self, other):
    #     return Arcimage(self.data % other.data)
    # def __rpow__(self, other):
    #     return Arcimage(self.data ** other.data)
    # def __rlshift__(self, other):
    #     return Arcimage(self.data << other.data)
    # def __rrshift__(self, other):
    #     return Arcimage(self.data >> other.data)
    # def __rand__(self, other):
    #     return Arcimage(self.data & other.data)
    # def __rxor__(self, other):
    #     return Arcimage(self.data ^ other.data)
    # def __ror__(self, other):
    #     return Arcimage(self.data | other.data)
    # def __iadd__(self, other):
    #     self.data += other.data
    #     return self
    # def __isub__(self, other):
    #     self.data -= other.data
    #     return self
    # def __imul__(self, other):
    #     self.data *= other.data
    #     return self
    # def __itruediv__(self, other):
    #     self.data /= other.data
    #     return self
    # def __ifloordiv__(self, other):
    #     self.data //= other.data
    #     return self
    # def __imod__(self, other):
    #     self.data %= other.data
    #     return self
    # def __ipow__(self, other):
    #     self.data **= other.data
    #     return self
    # def __ilshift__(self, other):
    #     self.data <<= other.data
    #     return self
    # def __irshift__(self, other):
    #     self.data >>= other.data
    #     return self
    # def __iand__(self, other):
    #     self.data &= other.data
    #     return self
    # def __ixor__(self, other):
    #     self.data ^= other.data
    #     return self
    # def __ior__(self, other):
    #     self.data |= other.data
    #     return self
    # def __neg__(self):
    #     return Arcimage(-self.data)
    # def __pos__(self):
    #     return Arcimage(+self.data)
    # def __abs__(self):
    #     return Arcimage(abs(self.data))
    # def __invert__(self):
    #     return Arcimage(~self.data)
    # def __complex__(self):
    #     return complex(self.data)

class Arcset:
    def __init__(self, dct):
        self.shots = len(dct['train'])
        self.imgs = []
        for abpair in dct['train']:
            self.imgs.append(Arcimage(np.array(abpair['input'])))
            self.imgs.append(Arcimage(np.array(abpair['output'])))
        for abpair in dct['test']:
            self.imgs.append(Arcimage(np.array(abpair['input'])))
            self.imgs.append(Arcimage(np.array(abpair['output'])))

class Arcsets:
    def __init__(self, dct=None):
        self.ids = []
        self.arcsets = []
        if dct is not None:
            for nm,dt in dct.items():
                self.ids.append(nm)
                self.arcsets.append(Arcset(dt))
    def __getitem__(self, key):
        return self.arcsets[self.ids.index(key)]
    def __len__(self):
        return len(self.ids)
    def __setitem__(self, key, value):
        self.ids.append(key)
        self.arcsets.append(value)
    def __iter__(self):
        return iter(zip(self.ids,self.arcsets))
    def __reversed__(self):
        return reversed(zip(self.ids,self.arcsets))
    @staticmethod
    def load_arcagi2(data_path): # data_path/0000.json
        train_files = os.listdir(data_path)
        training_data = {}
        for fn in train_files:
            sid = fn.strip(".json")
            with open(data_path + "/" + fn, 'r') as f:
                training_data[sid] = json.load(f)
        return Arcsets(training_data)
    @staticmethod
    def load_kaggle(data_path): # data_path/arc-agi_training_challenges.json
        with open(data_path + '/arc-agi_training_challenges.json', 'r') as f:
            training_data = json.load(f)
        with open(data_path + '/arc-agi_training_solutions.json', 'r') as f:
            train_sol = json.load(f)
        for k in training_data:
            for idx in range(len(training_data[k]['test'])):
                training_data[k]['test'][idx]['output'] = train_sol[k][idx]
        return Arcsets(training_data)
    

# "On the Measure of Intelligence" https://arxiv.org/pdf/1911.01547
# 1 Objectness priors: spatial or color/denoise/move-until-contact/grow-rebound
# 2 Goal-directedness prior: imagine time
# 3 Numbers and couting prior:
#   Many ARC tasks involve counting or sorting objects(e.g. sorting by size)
#   comparing numbers (e.g. which shape or symbol appears
#       the most/least/freq/largest/smallest/same size) or repetition.
#       addtion/substraction, all quantiies are smaller than approx. 10.
# 4 Basic geometry and topology prior:
#   Line, rectangular shapes (regular shapes are more likely than complex)
#   Symmetries, rotations, translations
#   Shape upscaling or downscaling, elastic distortions
#   Containing / being contained / being inside or ouside of a perimeter
#   Drawing lines, connecting points, orthogonal projections
#   Copying, repeating objects

class ObjectFeatures:
    def __init__(self):
        self.mask = 0
        self.features = [
            "framed",
            "isolated",
            "dominated color density", # float, ratio of pixels of dominated color            
            ]
        self.select_object_rule_description = [
            "unique color", # objects dominate color nunique counts argmin indices[0]
            ]
class Objectness:
    def __init__(self,img,ix,iy,nx,ny):
        self.features = ObjectFeatures()
        self.img_a = []
        for i in range(nx):
            row = [img[i+ix][u+iy] for u in range(ny)]
            self.img_a.append(row)
        self.img_b = img
        self.ix = ix
        self.iy = iy
        
        self.background_color = 0
        
        self.enclose_color=-1
        self.corner_color=-1
        self.mask = 0
        self.check()
    @staticmethod
    def mask2str(mask):
        s=''
        if mask & 1 > 0: s+="(bg-enclose)"
        if mask & 2 > 0: s+="(non-bg-enc)"
        if mask & 4 > 0: s+="(all-diff)"
        if mask & 8 > 0: s+="(corner-same)"
        if mask & 16 > 0: s+="(has-outside)"
        return s
    def check(self):
        img_a,img_b,ix,iy = self.img_a, self.img_b, self.ix, self.iy
        ra,ca = len(img_a),len(img_a[0])
        def is_pixel_framed(pa,pb,bg):
            return pb==bg #or pb!=pa
        local_mask = 0b00000111
        for idr in range(4):
            bval=None
            if idr==0: # top
                if ix>0:
                    bval = np.array([img_b[ix-1][iy+u] for u in range(ca)],dtype=int)
                    aval = np.array([img_a[0][u] for u in range(ca)],dtype=int)
            elif idr==1: # bottom
                if ix+ra<len(img_b): 
                    bval = np.array([img_b[ix+ra][iy+u] for u in range(ca)],dtype=int)
                    aval = np.array([img_a[ra-1][u] for u in range(ca)],dtype=int)
            elif idr==2: # left
                if iy>0:
                    bval = np.array([img_b[ix+u][iy-1] for u in range(ra)],dtype=int)
                    aval = np.array([img_a[u][0] for u in range(ra)],dtype=int)
            elif idr==3: # right
                if iy+ca<len(img_b[0]):
                    bval = np.array([img_b[ix+u][iy+ca] for u in range(ra)],dtype=int)
                    aval = np.array([img_a[u][0] for u in range(ra)],dtype=int)
            if not bval is None:
                if local_mask&1>0:
                    if not all([self.background_color==u for u in bval]):
                        local_mask = local_mask & 0b11111110
                if local_mask&2>0:
                    if self.enclose_color<0: self.enclose_color = bval[0]
                    if not all([self.enclose_color==u for u in bval]):
                        local_mask = local_mask & 0b11111101
                if local_mask&4>0:
                    if not all([bval[u]!=aval[u] for u in range(len(bval))]):
                        local_mask = local_mask & 0b11111011
        self.mask += local_mask
        if ix==0 or iy==0 or ix+ra==len(img_b) or iy+ca==len(img_b[0]):
            self.mask += 16
        else:
            c1 = img_b[ix-1][iy-1]
            c2 = img_b[ix-1][iy+ca]
            c3 = img_b[ix+ra][iy+ca]
            c4 = img_b[ix+ra][iy-1]
            if c1==c2 and c2==c3 and c3==c4:
                self.mask += 8
                self.corner_color = c1

    def validate(self):
        return self.good
def find_object(img, **kwargs):
    min_wh = kwargs.get("min_wh",2)
    nrow_min,nrow_max = min_wh,len(img)
    ncol_min,ncol_max = min_wh,len(img[0])
    objs=[]
    for h in range(nrow_min,nrow_max):
        for w in range(ncol_min,ncol_max):
            for ix in range(0,nrow_max-h+1):
                for iy in range(0,ncol_max-w+1):
                    obj = Objectness(img,ix,iy,h,w)
                    if obj.validate():
                        objs.append(obj)
                    subi = sub_img(img,ix,iy,h,w)
                    if is_object_like_region(img,ix,iy,h,w,background_color):
                        o = ObjectBlock(sub_img(img,ix,iy,h,w),background_color)
                        objs.append(o)
    return objs
def is_isolated(img_a, img_b, ix,iy,background_color=0):
    # check the adjacent pixel of img_a in img_b
    # exclude diagonal pixels, only 2*num_rows+2*num_cols
    # isolated pixel must be different, or must be background
    ra,ca = len(img_a),len(img_a[0])
    def is_pixel_framed(pa,pb,bg):
        return pb==bg #or pb!=pa
    dest = True
    if dest and ix>0: # top
        for j in range(ca):
            pixel_b = img_b[ix-1][iy+j]
            pixel_a = img_a[0][j]
            if is_pixel_framed(pixel_a,pixel_b,background_color):
                continue
            dest=False
            break
    if dest and iy>0: # left
        for i in range(ra):
            pixel_b = img_b[ix+i][iy-1]
            pixel_a = img_a[i][0]
            if is_pixel_framed(pixel_a,pixel_b,background_color):
                continue
            dest=False
            break
    if dest and ix+ra<len(img_b): # bottom
        for j in range(ca):
            pixel_b = img_b[ix+ra][iy+j]
            pixel_a = img_a[ra-1][j]
            if is_pixel_framed(pixel_a,pixel_b,background_color):
                continue
            dest=False
            break
    if dest and iy+ca<len(img_b[0]): # right
        for i in range(ra):
            pixel_b = img_b[ix+i][iy+ca]
            pixel_a = img_a[i][ca-1]
            if is_pixel_framed(pixel_a,pixel_b,background_color):
                continue
            dest=False
            break
    return dest

def is_incompressible(img,background_color=0):
    for i in range(len(img)):
        if all([img[i][u]==background_color for u in range(len(img[0]))]):
            return False
    for j in range(len(img[0])):
        if all([img[u][j]==background_color for u in range(len(img))]):
            return False
    return True
def sub_img(img,ix,iy,nx,ny):
    a = []
    for i in range(nx):
        row = [img[i+ix][u+iy] for u in range(ny)]
        a.append(row)
    return a
def is_object_like_region(img, ix,iy,nx,ny,background_color=0):
    # 1. sub-image isolated
    # 2. sub-image incompressible
    a = sub_img(img,ix,iy,nx,ny)
    if not is_isolated(a,img,ix,iy,background_color):
        return False
    if not is_incompressible(a,background_color):
        return False
    return True
    
