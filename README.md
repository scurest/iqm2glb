Python script to convert [Inter-Quake Model](http://sauerbraten.org/iqm/) `.iqm`
files to [glTF](https://www.khronos.org/gltf/) `.glb`.

Version support: IQM 2; glTF 2.0; GLB 2.

## Usage

### CLI

````
python iqm2glb.py INPUT.iqm [OUTPUT.glb]
````

If `OUTPUT.glb` is given, it will be over-written.

### Python

````py
import iqm2glb

with open('INPUT.iqm', 'rb') as f:
    iqm = f.read()

glb = iqm2glb.iqm2glb(iqm)
# glb.gltf contains the JSON
# glb.buffer contains the BIN
# You can modify them now if you like

with open('OUTPUT.glb', 'wb') as f:
    glb.write(f)
````

You can provide options with `iqm2glb(iqm, options)`. Look at `DEFAULT_OPTIONS`
in the source code for a list of available options.

The converter is a single file with no dependencies outside the standard
library, so just drop it wherever you want it.
