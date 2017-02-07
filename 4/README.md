==========================================================
                    INSTRUCTIONS
==========================================================

1. Ensure that you use the G++ compiler with version > 6.

2. Ensure all image files and the file 3.cpp reside within
   the same folder.

3. Execute
   g++ -g --std=c++14 `pkg-config --cflags --libs opencv` 3.cpp -lstdc++fc

4. Run the executable created by:
   ./a.out

5. Adjust the trackbars in the GUI to change the image
   and filter specifications.
