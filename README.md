# YetAnotherArcSolver
Solve The Abstraction and Reasoning Corpus for Artificial General Intelligence

## NotebookLM notes:

Based on the sources, there are numerous elementary operations used in Domain-Specific Languages (DSLs) designed for solving Abstraction and Reasoning Corpus (ARC) tasks. These operations form the building blocks for programs that aim to transform input grids into output grids. Here's a summary of these operations, categorized for clarity:

**I. Rigid Transformations:** These operations involve geometric transformations without changing the shape or size of objects, only their orientation or position (implicitly, as they operate on the entire grid or selections within it).

*   **Rotations:** Rotate the grid by 90°, 180°, or 270° clockwise or counter-clockwise.
*   **Flips:** Flip the grid horizontally or vertically.
*   **Transposition:** Swap the x and y axes (transpose the grid).

**II. Cropping and Padding/Expansion:** These operations modify the dimensions of the grid by removing parts or adding blank spaces or repetitions.

*   **Cropping:** Crop the left half, right half, top half, or bottom half of the grid.
*   **Slicing:** Slice grids along an axis.
*   **Repeating/Mirroring:** Repeat or mirror the grid to create larger symmetrical patterns.
*   **Upscaling/Downscaling:** Multiply grid size and expand pixels accordingly.
*   **Resizing:** Change the dimensions of the grid.
*   **Uncropping:** Reverse a cropping operation (requires knowing the original size/position).

**III. Color Manipulation:** These operations involve changing the colors of pixels or objects within the grid.

*   **Color Replacement/Update:** Change the color of a specific pixel or all pixels of a given color to another color.
*   **Color Swapping:** Swap the colors of two specified nodes or all instances of two colors in the grid.
*   **Color Copying:** Copy the color of one node or object to another.
*   **Color Filtering/Erasing:** Remove or make transparent pixels of a specific color.
*   **Color Remapping:** Change one set of colors to another set.
*   **Identifying Colors:** Operations that target the most or least frequent color.
*   **Color Blending:** Combine colors based on some rule.

**IV. Object Manipulation:** These operations work on identified "objects" within the grid. The definition of an object can vary (e.g., connected pixels of the same color).

*   **Object Detection/Extraction:** Identify and isolate individual objects based on connectivity (4-connected or 8-connected) and color.
*   **Object Movement:** Move an object to a specific location, boundary of another object, or in a given direction by a certain step.
*   **Object Extension/Expansion:** Extend an object until it hits another object or the grid boundary in a specified direction.
*   **Object Filling:** Fill a subgrid or the inside of a shape with a specific object or color.
*   **Object Hollowing:** Create a hollow version of an object.
*   **Object Mirroring/Reflection:** Reflect an object across a horizontal or vertical axis.
*   **Object Insertion/Duplication:** Insert a new object or duplicate an existing one.
*   **Object Connection:** Connect two pixels or objects with a line of a specified color.
*   **Object Truncation/Removal:** Remove a specified object and potentially recolor the vacated pixels.
*   **Object Sorting:** Arrange objects based on attributes like size.

**V. Combining and Splitting Grids/Objects:** Operations that create new grids or lists of grids from existing ones.

*   **Grid Stacking/Composition:** Stack grids in an arbitrary order, optionally considering original positions.
*   **Pixelwise Operations:** Perform logical operations (e.g., AND, OR) on corresponding pixels of two grids.
*   **Grid Splitting:** Split a grid into a list of sub-grids based on color, connected components, rows, or columns.

**VI. Basic Arithmetic and Counting:** Operations involving numerical aspects of the grid.

*   **Counting:** Count the number of pixels of a specific color, total number of pixels, or the number of connected objects.
*   **Size Comparison:** Determine the largest, smallest, odd, or even sized objects.
*   **Frequency Analysis:** Identify the most or least frequently occurring colors.

**VII. Spatial Relations:** Operations that leverage or modify spatial relationships between elements.

*   **Identifying Neighbors:** Determine adjacency (4-connected or 8-connected) between pixels or objects.
*   **Relative Positioning:** Determine if one node is to the left, right, above, or below another.
*   **Containment:** Determine if one object contains or partially contains another.
*   **Contact/Collision:** Operations that relate to objects touching each other.

**VIII. Pattern-Based Operations:** These involve recognizing and manipulating patterns within the grid.

*   **Pattern Detection:** Identify recurring arrangements of colors or shapes.
*   **Pattern Copying/Pasting:** Replicate detected patterns to other parts of the grid.
*   **Inverse Patterns:** Generate the opposite of an existing pattern.
*   **Scaling Patterns:** Enlarge or shrink patterns.

**IX. Control Flow and Logical Operations (within Programs):** While not strictly "elementary grid operations," these are crucial for constructing more complex transformations within a program.

*   **Conditional Logic:** Perform different operations based on certain conditions (e.g., presence of a color, size of an object).
*   **Iteration/Looping:** Repeat a sequence of operations multiple times.
*   **Function Calls:** Use predefined subroutines or primitives.

It's important to note that the specific set of elementary operations varies depending on the DSL or program synthesis framework being used. The design of the DSL is a critical factor in the success of these approaches, requiring a balance between expressiveness and the tractability of the search space. Furthermore, some advanced frameworks might use higher-level actions that are composed of sequences of these more elementary operations.
