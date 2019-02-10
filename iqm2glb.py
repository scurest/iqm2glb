#!/usr/bin/env python3
"""
Convert IQM 2 to glTF 2.0 GLB.
License: Public Domain
"""
import struct, json, os, sys, logging, time

logging.getLogger(__name__).addHandler(logging.NullHandler())

DEFAULT_OPTIONS = {
    # Model name.
    'model_name': '',
    # Character encoding for text suitable for Python's bytes.decode().
    'character_encoding': 'utf-8',
    # When an IQM material name looks like an image path (ex. because it ends
    # with .png), use it as the path for a baseColorTexture for that material.
    'guess_texture_names': True,
    # Include animations, if present.
    'include_animations': True,
    # Include the adjacency information, if present, in an accessor referenced
    # by mesh['extras']['iqm_adjacency'].
    'include_adjacency': True,
    # Include bounding box information, if present, in animation samplers
    # referenced by
    # animation['extras']['iqm_bounds']['bbmin'/'bbmax'/'xyradius'/'radius'].
    'include_bounds': True,
}

IQM_POSITION     = 0
IQM_TEXCOORD     = 1
IQM_NORMAL       = 2
IQM_TANGENT      = 3
IQM_BLENDINDEXES = 4
IQM_BLENDWEIGHTS = 5
IQM_COLOR        = 6
IQM_CUSTOM       = 0x10

IQM_BYTE   = 0
IQM_UBYTE  = 1
IQM_SHORT  = 2
IQM_USHORT = 3
IQM_INT    = 4
IQM_UINT   = 5
IQM_HALF   = 6
IQM_FLOAT  = 7
IQM_DOUBLE = 8

IQM_LOOP = 1

IQM_HEADER      = struct.Struct('<25I')  # minus magic/version/filesize
IQM_MESH        = struct.Struct('<6I')
IQM_TRIANGLE    = struct.Struct('<3I')
IQM_JOINT       = struct.Struct('<Ii10f')
IQM_POSE        = struct.Struct('<iI20f')
IQM_ANIMATION   = struct.Struct('<3IfI')
IQM_VERTEXARRAY = struct.Struct('<5I')
IQM_BOUNDS      = struct.Struct('<8f')

GLTF_BYTE           = 5120
GLTF_UNSIGNED_BYTE  = 5121
GLTF_SHORT          = 5122
GLTF_UNSIGNED_SHORT = 5123
GLTF_UNSIGNED_INT   = 5125
GLTF_FLOAT          = 5126


class Glb:
    def __init__(self, gltf, buffer):
        self.gltf = gltf
        self.buffer = buffer

    def write(self, w):
        """Write the GLB to `w`, a `write()`-supporting file-like object."""
        json_data = bytearray(
            json.dumps(self.gltf, indent=None, separators=(',', ':')),
            encoding='utf-8',
        )
        _pad_to_alignment(json_data, 4, b' ')
        buffer_padding = (4 - (len(self.buffer) % 4)) % 4
        filesize = 4*3 + 4*2+len(json_data) + 4*2+len(self.buffer)+buffer_padding

        w.write(b'glTF\02\00\00\00')
        w.write(struct.pack('<2I', filesize, len(json_data)))
        w.write(b'JSON')
        w.write(json_data)
        w.write(struct.pack('<I', len(self.buffer) + buffer_padding))
        w.write(b'BIN\0')
        w.write(self.buffer)
        w.write(b'\0' * buffer_padding)


def iqm2glb(iqm, options={}):
    """
    Convert `iqm`, a bytes-like object containing an IQM file, to `Glb`.
    """
    start_time = time.time_ns()
    logger = logging.getLogger(__name__)
    options = {**DEFAULT_OPTIONS, **options}

    magic, = struct.unpack_from('<16s', iqm, offset=0)
    if magic != b'INTERQUAKEMODEL\0':
        raise Exception('not an IQM file; wrong magic number')

    version, = struct.unpack_from('<I', iqm, offset=16)
    if version != 2:
        raise Exception(f'unsupported IQM version {version} (must be 2)')

    filesize, = struct.unpack_from('<I', iqm, offset=20)
    if filesize != len(iqm):
        raise Exception(f'wrong filesize (reported: {filesize}, actual: {len(iqm)})')

    flags, \
    __num_text, ofs_text, \
    num_meshes, ofs_meshes, \
    num_vertexarrays, num_vertexes, ofs_vertexarrays, \
    num_triangles, ofs_triangles, ofs_adjacency, \
    num_joints, ofs_joints, \
    num_poses, ofs_poses, \
    num_anims, ofs_anims, \
    num_frames, num_framechannels, ofs_frames, ofs_bounds, \
    num_comment, ofs_comment, \
    __num_extensions, __ofs_extensions = \
        IQM_HEADER.unpack_from(iqm, offset=24)

    def get_string(ofs):
        if ofs == 0: return ''
        start = ofs_text + ofs
        end = iqm.find(b'\0', start)
        return iqm[start:end].decode(options['character_encoding'], errors='replace')

    gltf = {'asset': {'version': '2.0'}}
    buffer = bytearray()

    if ofs_comment and num_comment:
        comments = []
        ofs = ofs_comment
        while len(comments) < num_comment:
            end = iqm.find(b'\0', ofs)
            comments.append(
                bytes(iqm[ofs:end]).decode(options['character_encoding'], errors='replace')
            )
            ofs = end + 1
        gltf.setdefault('extras', {})['iqm_comments'] = comments

    # Turn vertex arrays into accessors
    if ofs_vertexarrays and num_vertexarrays and num_vertexes:
        vertex_arrays = {}  # maps vertexarray types to accessor idxs
        for i in range(0, num_vertexarrays):
            vertex_array_type, flags, vertex_array_format, size, offset = \
                IQM_VERTEXARRAY.unpack_from(iqm, offset=ofs_vertexarrays + i*IQM_VERTEXARRAY.size)
            if vertex_array_type in vertex_arrays:
                raise Exception(f'multiple vertex arrays of type {vertex_array_type}')
            if vertex_array_format == IQM_INT:
                logger.warning(
                    f'skipping vertex array (type: {vertex_array_type}); '
                    "glTF doesn't support INT format"
                )
                continue
            gltf_type = dict([
                (1, 'SCALAR'),
                (2, 'VEC2'),
                (3, 'VEC3'),
                (4, 'VEC4'),  # could also be MAT2. Oh well.
                (9, 'MAT3'),
                (16, 'MAT4'),
            ]).get(size)
            if not gltf_type:
                logger.warning(
                    f'skipping vertex array (type: {vertex_array_type}); '
                    f"don't know what to do with elements with {size} components"
                )
                continue
            gltf_component_type = dict([
                (IQM_BYTE, GLTF_BYTE),
                (IQM_UBYTE, GLTF_UNSIGNED_BYTE),
                (IQM_SHORT, GLTF_UNSIGNED_SHORT),
                (IQM_USHORT, GLTF_SHORT),
                (IQM_UINT, GLTF_UNSIGNED_INT),
                (IQM_HALF, GLTF_FLOAT),
                (IQM_FLOAT, GLTF_FLOAT),
                (IQM_DOUBLE, GLTF_FLOAT),
            ])[vertex_array_format]
            normalized = \
                gltf_component_type != GLTF_FLOAT and \
                vertex_array_type != IQM_BLENDINDEXES and \
                vertex_array_type < IQM_CUSTOM  # XXX: should custom types be normalized?
            fmt = dict([
                (IQM_BYTE, 'b'),
                (IQM_UBYTE, 'B'),
                (IQM_SHORT, 'h'),
                (IQM_USHORT, 'H'),
                (IQM_UINT, 'I'),
                (IQM_HALF, 'e'),
                (IQM_FLOAT, 'f'),
                (IQM_DOUBLE, 'd'),
            ])[vertex_array_format]
            elem_size = size * struct.calcsize(fmt)

            # glTF wants min/max of vertex positions
            if vertex_array_type == IQM_POSITION:
                unpack_from = struct.Struct(f'<{size}{fmt}').unpack_from
                stride = struct.calcsize(f'<{size}{fmt}')
                min_pos = list(unpack_from(iqm, offset))
                max_pos = list(min_pos)
                for i in range(1, num_vertexes):
                    x = unpack_from(iqm, offset=offset + i*stride)
                    for j in range(0, size):
                        min_pos[j] = min(min_pos[j], x[j])
                        max_pos[j] = max(max_pos[j], x[j])

            if elem_size % 4 == 0 and vertex_array_format not in [IQM_HALF, IQM_DOUBLE]:
                # Straight-copy the data from iqm into buffer
                _pad_to_alignment(buffer, 4)
                byte_offset = len(buffer)
                buffer += iqm[offset:offset + elem_size*num_vertexes]
            else:
                # Need to repack
                # TODO: untested
                input_fmt = struct.Struct(f'<{size}{fmt}')
                if fmt == 'd':
                    logger.info(
                        'repacking vertex array (type: {vertex_array_type}) stored as DOUBLEs '
                        'as FLOATs; this loses precision!'
                    )
                if fmt == 'e' or 'd':
                    output_fmt = f'<{size}f'
                else:
                    output_fmt = f'<{size}{fmt}'
                if struct.calcsize(output_fmt) % 4 != 0:
                    output_fmt += (4 - (struct.calcsize(output_fmt) % 4)) * 'x'
                output_fmt = struct.Struct(output_fmt)

                _pad_to_alignment(buffer, 4)
                byte_offset = len(buffer)
                inputs = input_fmt.iter_unpack(iqm[offset:offset + input_fmt.size*num_vertexes])
                for val in inputs:
                    buffer += output_fmt.pack(*val)

            gltf.setdefault('bufferViews', []).append({
                'buffer': 0,
                'byteOffset': byte_offset,
                'byteLength': len(buffer) - byte_offset,
            })
            accessor = {
                'bufferView': len(gltf['bufferViews']) - 1,
                'type': gltf_type,
                'componentType': gltf_component_type,
                'count': num_vertexes,
            }
            if normalized: accessor['normalized'] = True
            if vertex_array_type == IQM_POSITION:
                accessor['min'] = min_pos
                accessor['max'] = max_pos
            gltf.setdefault('accessors', []).append(accessor)

            vertex_arrays[vertex_array_type] = len(gltf['accessors']) - 1

    # Make index accessors
    if ofs_triangles and num_triangles:
        if num_vertexes <= 0xffff:
            # Repack as USHORTs
            index_size = 2
            _pad_to_alignment(buffer, 2)
            offset = len(buffer)
            for i in range(0, num_triangles):
                tri = IQM_TRIANGLE.unpack_from(iqm, offset=ofs_triangles + i*3*4)
                buffer += struct.pack('<3H', *tri)
        else:
            index_size = 4
            _pad_to_alignment(buffer, 4)
            offset = len(buffer)
            buffer += iqm[ofs_triangles:ofs_triangles + 3 * 4 * num_triangles]

        gltf.setdefault('bufferViews', {}).append({
            'buffer': 0,
            'byteOffset': offset,
            'byteLength': len(buffer) - offset,
        })
        index_bufferview = len(gltf['bufferViews']) - 1

    # Create the mesh (one glTF primitive = one IQM mesh)
    if ofs_meshes and num_meshes:
        primitives = []
        material_names = []
        for i in range(0, num_meshes):
            name, material, __first_vertex, __num_vertices, first_triangle, num_triangles = \
                IQM_MESH.unpack_from(iqm, offset=ofs_meshes + i*IQM_MESH.size)

            primitive = {}
            if name:
                primitive.setdefault('extras', {})['name'] = get_string(name)
            if material:
                material_name = get_string(material)
                if material_name not in material_names:
                    material_names.append(material_name)
                    primitive['material'] = len(material_names) - 1
                else:
                    primitive['material'] = material_names.index(material_name)

            for ty, accessor_idx in vertex_arrays.items():
                if ty == IQM_POSITION: attr_name = 'POSITION'
                elif ty == IQM_TEXCOORD: attr_name = 'TEXCOORD_0'
                elif ty == IQM_NORMAL: attr_name = 'NORMAL'
                elif ty == IQM_TANGENT: attr_name = 'TANGENT'
                elif ty == IQM_BLENDWEIGHTS: attr_name = 'WEIGHTS_0'
                elif ty == IQM_BLENDINDEXES: attr_name = 'JOINTS_0'
                elif ty == IQM_COLOR: attr_name = 'COLOR_0'
                else: attr_name = '_' + get_string(ty - IQM_CUSTOM)
                primitive.setdefault('attributes', {})[attr_name] = accessor_idx

            gltf.setdefault('accessors', {}).append({
                'bufferView': index_bufferview,
                'type': 'SCALAR',
                'componentType': GLTF_UNSIGNED_SHORT if index_size == 2 else GLTF_UNSIGNED_INT,
                'byteOffset': index_size * 3 * first_triangle,
                'count': 3 * num_triangles,
            })
            primitive['indices'] = len(gltf['accessors']) - 1

            primitives.append(primitive)

        gltf['meshes'] = [{'primitives': primitives}]

        if options['model_name']:
            gltf['meshes'][0]['name'] = options['model_name']

        # Create materials
        if material_names:
            gltf['materials'] = [{'name': name} for name in material_names]
            if options['guess_texture_names']:
                # If the material name looks like an image path, use that image for it
                texture_names = list(set(name for name in material_names if _is_image_path(name)))
                for idx, name in enumerate(material_names):
                    if _is_image_path(name):
                        gltf['materials'][idx]['pbrMetallicRoughness'] = {
                            'baseColorTexture': {'index': texture_names.index(name)},
                            'metallicFactor': 0,
                        }
                if texture_names:
                    gltf['textures'] = [{'source': idx} for idx, __name in enumerate(texture_names)]
                    gltf['images'] = [{'uri': name} for name in texture_names]

        # Adjaceny
        if ofs_adjacency and options['include_adjacency']:
            _pad_to_alignment(buffer, 4)
            byte_offset = len(buffer)
            buffer += iqm[ofs_adjacency:ofs_adjacency + 4*3*num_triangles]
            gltf.setdefault('bufferViews', {}).append({
                'buffer': 0,
                'byteOffset': byte_offset,
                'byteLength': len(buffer) - byte_offset,
            })
            gltf.setdefault('accessors', {}).append({
                'bufferView': len(gltf['bufferViews']) - 1,
                'type': 'SCALAR',
                'componentType': GLTF_UNSIGNED_INT,
                'count': 3 * num_triangles,
            })
            gltf['meshes'][0].setdefault('extras', {})['iqm_adjacency'] = len(gltf['accessors']) - 1

    # Create nodes
    if ofs_joints and num_joints:
        nodes = [{} for i in range(0, num_joints)]
        parents = [None] * num_joints  # parent's index for nth node
        parent_to_local_mats = [None] * num_joints
        roots = []  # roots of node forest

        for i in range(0, num_joints):
            name, parent, tx, ty, tz, qx, qy, qz, qw, sx, sy, sz = \
                IQM_JOINT.unpack_from(iqm, offset=ofs_joints + i*IQM_JOINT.size)
            if name:
                nodes[i]['name'] = get_string(name)
            if parent >= 0:
                nodes[parent].setdefault('children', []).append(i)
                parents[i] = parent
            else:
                roots.append(i)
            if [tx, ty, tz] != [0, 0, 0]:
                nodes[i]['translation'] = [tx, ty, tz]
            nodes[i]['rotation'] = [qx, qy, qz, qw]
            if [sx, sy, sz] != [1, 1, 1]:
                nodes[i]['scale'] = [sx, sy, sz]

            parent_to_local_mats[i] = _inverse_trs(tx, ty, tz, qx, qy, qz, qw, sx, sy, sz)

        gltf['nodes'] = nodes

        # Create skin
        needs_skin = \
            IQM_BLENDINDEXES in vertex_arrays and \
            IQM_BLENDWEIGHTS in vertex_arrays
        if needs_skin:
            skin = {'joints': list(range(0, num_joints))}
            if len(roots) == 1:
                skin['skeleton'] = roots[0]

            inv_bind_mats = [None] * num_joints
            def compute_inv_bind_mat(i):
                if inv_bind_mats[i] is None:
                    if parents[i] is None:
                        inv_bind_mats[i] = parent_to_local_mats[i]
                    else:
                        parent_inv_bind = compute_inv_bind_mat(parents[i])
                        inv_bind_mats[i] = _mul_mat3x4(parent_to_local_mats[i], parent_inv_bind)
                return inv_bind_mats[i]

            _pad_to_alignment(buffer, 4)
            offset = len(buffer)
            for i in range(0, num_joints):
                m = compute_inv_bind_mat(i)
                buffer += struct.pack('<16f',
                    *m[0:3], 0, *m[3:6], 0, *m[6:9], 0, *m[9:12], 1,
                )
            gltf.setdefault('bufferViews', []).append({
                'buffer': 0,
                'byteOffset': offset,
                'byteLength': len(buffer) - offset,
            })
            gltf.setdefault('accessors', []).append({
                'bufferView': len(gltf['bufferViews']) - 1,
                'type': 'MAT4',
                'componentType': GLTF_FLOAT,
                'count': num_joints,
            })
            skin['inverseBindMatrices'] = len(gltf['accessors']) - 1

            gltf['skins'] = [skin]

        # Add a new node to instantiate the mesh/skin at
        gltf['nodes'].append({'mesh': 0})
        if needs_skin:
            gltf['nodes'][-1]['skin'] = 0
        if options['model_name']:
            gltf['nodes'][-1]['name'] = options['model_name']

    if ofs_anims and num_anims and options['include_animations']:
        # constant_trs[joint_idx][trs] contains the value of that TRS property
        # for that joint when it is constant over all time
        constant_trs = []
        for joint_idx in range(0, num_poses):
            parent, channel_mask, *rest = \
                IQM_POSE.unpack_from(iqm, offset=ofs_poses + joint_idx*IQM_POSE.size)
            channel_offset = rest[:10]
            trs = {}
            if channel_mask & 0b0000000111 == 0:
                trs['translation'] = channel_offset[0:3]
            if channel_mask & 0b0001111000 == 0:
                trs['rotation'] = _normalize_quat(channel_offset[3:7])
            if channel_mask & 0b1110000000 == 0:
                trs['scale'] = channel_offset[7:10]
            constant_trs.append(trs)

        # animated_trs[frame][joint_idx][trs] contains the value of that TRS
        # property for that joint at that frame when it depends on the frame
        num_framedata = num_framechannels * num_frames
        framedata = struct.unpack_from(f'<{num_framedata}H', iqm, offset=ofs_frames)
        framedata_idx = 0
        animated_trs = []
        for __frame_idx in range(0, num_frames):
            frame = []
            for joint_idx in range(0, num_poses):
                parent, channel_mask, *rest = \
                    IQM_POSE.unpack_from(iqm, offset=ofs_poses + joint_idx*IQM_POSE.size)
                channel_offset = rest[:10]
                channel_scale = rest[10:20]
                for i in range(0, 10):
                    if channel_mask & (1 << i):
                        channel_offset[i] += framedata[framedata_idx] * channel_scale[i]
                        framedata_idx += 1
                trs = {}
                if channel_mask & 0b0000000111:
                    trs['translation'] = channel_offset[0:3]
                if channel_mask & 0b0001111000:
                    trs['rotation'] = _normalize_quat(channel_offset[3:7])
                if channel_mask & 0b1110000000:
                    trs['scale'] = channel_offset[7:10]
                frame.append(trs)
            animated_trs.append(frame)

        # We're going to write out animation data
        # * One bufferview for all input/output data
        # * Constant joint-path pairs get a sampler with a single keyframe with
        #   their constant value
        # * Reserve input accessors as we go, but defer actually writing them
        #   until the end so we can share buffer data between them
        # * share the buffer data/accessors for constant TRS properties across
        #   all animations
        class AnimData:
            pass
        ad = AnimData()

        # Store data in this temporary buffer and copy it to the main buffer at the end
        ad.buffer = bytearray()

        # Index of the buffer view reserved for all animation data
        ad.bufview = None
        def get_anim_bufview():
            if ad.bufview is None:
                gltf.setdefault('bufferViews', []).append({})
                ad.bufview = len(gltf['bufferViews']) - 1
            return ad.bufview

        # Maps a framerate/num_frames pair to the index of the accessor reserved for it
        ad.timeline_accessors = {}
        def get_timeline_accessor(framerate, num_frames):
            if (framerate, num_frames) not in ad.timeline_accessors:
                gltf.setdefault('accessors', []).append({})
                ad.timeline_accessors[(framerate, num_frames)] = len(gltf['accessors']) - 1
            return ad.timeline_accessors[(framerate, num_frames)]

        ad.constant_input_accessor = None
        ad.constant_output_accessors = {}
        # Gets a sampler input/output accessor pair for the constant joint-path pairs
        # Cached because it's the same for all animations
        def get_constant_input_output(joint_idx, path):
            if ad.constant_input_accessor is None:
                byte_offset = len(ad.buffer)
                ad.buffer += struct.pack('<f', 0)
                gltf.setdefault('accessors', []).append({
                    'bufferView': get_anim_bufview(),
                    'type': 'SCALAR',
                    'componentType': GLTF_FLOAT,
                    'count': 1,
                    'byteOffset': byte_offset,
                    'min': [0],
                    'max': [0],
                })
                ad.constant_input_accessor = len(gltf['accessors']) - 1

            if (joint_idx, path) not in ad.constant_output_accessors:
                byte_offset = len(ad.buffer)
                num_components = 4 if path == 'rotation' else 3
                ad.buffer += struct.pack(f'<{num_components}f', *constant_trs[joint_idx][path])
                gltf.setdefault('accessors', []).append({
                    'bufferView': get_anim_bufview(),
                    'type': 'VEC4' if path == 'rotation' else 'VEC3',
                    'componentType': GLTF_FLOAT,
                    'count': 1,
                    'byteOffset': byte_offset,
                })
                ad.constant_output_accessors[(joint_idx, path)] = len(gltf['accessors']) - 1

            return (ad.constant_input_accessor, ad.constant_output_accessors[(joint_idx, path)])

        animations = []
        for anim_idx in range(0, num_anims):
            name, first_frame, num_frames, framerate, flags = \
                IQM_ANIMATION.unpack_from(iqm, offset=ofs_anims + anim_idx*IQM_ANIMATION.size)
            if num_frames == 0:
                logger.info(
                    'skipping animation with no frames '
                    f'(index: {anim_idx}, name: {get_string(name)})'
                )
                continue
            loop = flags & IQM_LOOP

            anim = {}
            if name:
                anim['name'] = get_string(name)
            frame_count = num_frames
            if loop:
                anim.setdefault('extras', {})['iqm_loop'] = True
                frame_count += 1  # See comment about looping below
            channels = anim['channels'] = []
            samplers = anim['samplers'] = []
            for joint_idx in range(0, num_poses):
                for path in ['translation', 'rotation', 'scale']:
                    if path in constant_trs[joint_idx]:
                        input_accessor, output_accessor = \
                            get_constant_input_output(joint_idx, path)
                    else:
                        input_accessor = get_timeline_accessor(framerate, frame_count)

                        num_components = 4 if path == 'rotation' else 3
                        byte_offset = len(ad.buffer)
                        fmt = struct.Struct(f'<{num_components}f')
                        for i in range(first_frame, first_frame + num_frames):
                            ad.buffer += fmt.pack(*animated_trs[i][joint_idx][path])
                        # When we loop a continuous-time curve, when we reach the
                        # last sample we jump instantly to the first sample again.
                        # But in frame-based curves, there is a time interval of
                        # length 1/framerate between the last sample and the
                        # reoccurence of the first. To preserve this effect, looping
                        # animations have their first frame duplicated at the end.
                        #
                        # (This is easy to understand if you draw the graphs.)
                        if loop:
                            ad.buffer += fmt.pack(*animated_trs[first_frame][joint_idx][path])

                        gltf.setdefault('accessors', []).append({
                            'bufferView': get_anim_bufview(),
                            'type': 'VEC4' if path == 'rotation' else 'VEC3',
                            'componentType': GLTF_FLOAT,
                            'count': frame_count,
                            'byteOffset': byte_offset,
                        })
                        output_accessor = len(gltf['accessors']) - 1

                    samplers.append({
                        'input': input_accessor,
                        'output': output_accessor,
                    })
                    channels.append({
                        'target': {'node': joint_idx, 'path': path},
                        'sampler': len(samplers) - 1,
                    })

            # Bounding boxes
            if ofs_bounds and options['include_bounds']:
                bbmins, bbmaxes, xyradii, radii = [], [], [], []
                for frame_idx in range(first_frame, first_frame + num_frames):
                    bounds = IQM_BOUNDS.unpack_from(iqm, offset=ofs_bounds + frame_idx*IQM_BOUNDS.size)
                    bbmins.append(bounds[0:3])
                    bbmaxes.append(bounds[3:6])
                    xyradii.append([bounds[6]])
                    radii.append([bounds[7]])
                input_accessor = get_timeline_accessor(framerate, frame_count)
                iqm_bounds = {}
                for name, num_components, values in [
                    ('bbmin', 3, bbmins),
                    ('bbmax', 3, bbmaxes),
                    ('xyradius', 1, xyradii),
                    ('radius', 1, radii)
                ]:
                    byte_offset = len(ad.buffer)
                    fmt = struct.Struct('<' + str(num_components) + 'f')
                    for val in values:
                        ad.buffer += fmt.pack(*val)
                    if loop:
                        ad.buffer += fmt.pack(*values[0])
                    gltf.setdefault('accessors', []).append({
                        'bufferView': get_anim_bufview(),
                        'type': 'SCALAR' if num_components == 1 else 'VEC3',
                        'componentType': GLTF_FLOAT,
                        'count': frame_count,
                        'byteOffset': byte_offset,
                    })
                    output_accessor = len(gltf['accessors']) - 1
                    samplers.append({
                        'input': input_accessor,
                        'output': output_accessor,
                    })
                    iqm_bounds[name] = len(samplers) - 1
                anim.setdefault('extras', {})['iqm_bounds'] = iqm_bounds

            animations.append(anim)

        if animations:
            gltf['animations'] = animations

            # Finish up all the accessor/bufferViews we reserved but didn't fill
            # in

            # Write one list of floats for each framerate long enough for the
            # longest animation that used that framerate

            framerate_to_max_num_frames = {}
            for framerate, num_frames in ad.timeline_accessors:
                framerate_to_max_num_frames.setdefault(framerate, 0)
                framerate_to_max_num_frames[framerate] = max(
                    framerate_to_max_num_frames[framerate],
                    num_frames,
                )

            framerate_to_byte_offset = {}
            for framerate, num_frames in framerate_to_max_num_frames.items():
                framerate_to_byte_offset[framerate] = len(ad.buffer)
                for i in range(0, num_frames):
                    ad.buffer += struct.pack('<f', i / framerate)

            for (framerate, num_frames), accessor_idx in ad.timeline_accessors.items():
                gltf['accessors'][accessor_idx] = {
                    'bufferView': ad.bufview,
                    'type': 'SCALAR',
                    'componentType': GLTF_FLOAT,
                    'count': num_frames,
                    'byteOffset': framerate_to_byte_offset[framerate],
                    'min': [0],
                    'max': [(num_frames - 1) / framerate],
                }

            _pad_to_alignment(buffer, 4)
            byte_offset = len(buffer)
            buffer += ad.buffer
            gltf['bufferViews'][ad.bufview] = {
                'buffer': 0,
                'byteOffset': byte_offset,
                'byteLength': len(ad.buffer),
            }

    gltf['buffers'] = [{'byteLength': len(buffer)}]

    elapsed_time = (time.time_ns() - start_time) / 1e9
    logger.info('Conversion finished in %.3fs', elapsed_time)

    return Glb(gltf, buffer)


def _is_image_path(name):
    dot_pos = name.rfind('.')
    if dot_pos == -1: return False
    ext = name[dot_pos + 1:].lower()
    return ext in [
        'png', 'jpg', 'jpeg',
        'tga', 'dds',  # non-standard in glTF
    ]


def _mul_mat3x4(m1, m2):
    # A 3x4 matrix is treated as a 4x4 matrix with final row (0 0 0 1).
    # Matrices are column-major.
    return [
        m1[0]*m2[0] + m1[3]*m2[1] + m1[6]*m2[2],
        m1[1]*m2[0] + m1[4]*m2[1] + m1[7]*m2[2],
        m1[2]*m2[0] + m1[5]*m2[1] + m1[8]*m2[2],

        m1[0]*m2[3] + m1[3]*m2[4] + m1[6]*m2[5],
        m1[1]*m2[3] + m1[4]*m2[4] + m1[7]*m2[5],
        m1[2]*m2[3] + m1[5]*m2[4] + m1[8]*m2[5],

        m1[0]*m2[6] + m1[3]*m2[7] + m1[6]*m2[8],
        m1[1]*m2[6] + m1[4]*m2[7] + m1[7]*m2[8],
        m1[2]*m2[6] + m1[5]*m2[7] + m1[8]*m2[8],

        m1[0]*m2[9] + m1[3]*m2[10] + m1[6]*m2[11] + m1[9],
        m1[1]*m2[9] + m1[4]*m2[10] + m1[7]*m2[11] + m1[10],
        m1[2]*m2[9] + m1[5]*m2[10] + m1[8]*m2[11] + m1[11],
    ]


def _quaternion_to_mat3x4(qx, qy, qz, qw):
    qx2 = 2 * qx ; qy2 = 2 * qy ; qz2 = 2 * qz
    qxx2 = qx2 * qx ; qxy2 = qx2 * qy ; qxz2 = qx2 * qz
    qyy2 = qy2 * qy ; qyz2 = qy2 * qz ; qzz2 = qz2 * qz
    qwy2 = qy2 * qw ; qwz2 = qz2 * qw ; qwx2 = qx2 * qw
    return [
        1 - qyy2 - qzz2, qxy2 + qwz2, qxz2 - qwy2,
        qxy2 - qwz2, 1 - qxx2 - qzz2, qyz2 + qwx2,
        qxz2 + qwy2, qyz2 - qwx2, 1 - qxx2 - qyy2,
        0, 0, 0,
    ]


def _inverse_trs(tx, ty, tz, qx, qy, qz, qw, sx, sy, sz):
    inv_trans = [1, 0, 0, 0, 1, 0, 0, 0, 1, -tx, -ty, -tz]
    inv_rot = _quaternion_to_mat3x4(-qx, -qy, -qz, qw)
    inv_scale = [1/sx, 0, 0, 0, 1/sy, 0, 0, 0, 1/sz, 0, 0, 0]
    return _mul_mat3x4(inv_scale, _mul_mat3x4(inv_rot, inv_trans))


def _pad_to_alignment(buffer, alignment, char=b'\0'):
    padding_amt = (alignment - (len(buffer) % alignment)) % alignment
    buffer += padding_amt * char


def _normalize_quat(q):
    x, y, z, w = q
    norm = (x*x + y*y + z*z + w*w)**0.5
    if norm == 0: return q
    return [x/norm, y/norm, z/norm, w/norm]


if __name__ == '__main__':
    if len(sys.argv) not in [2, 3] or sys.argv[1] in ['-h', '--help']:
        print('usage: iqm2glb input.iqm [output.glb]')
        sys.exit(1)

    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    input_path = sys.argv[1]
    if len(sys.argv) == 3:
        output_path = sys.argv[2]
    else:
        filename = os.path.basename(input_path)
        if filename.endswith('.iqm') or filename.endswith('.IQM'):
            filename = filename[:-4]
        output_path = filename + '.glb'
        i = 1
        while os.path.exists(output_path):
            output_path = filename + f'.{i}.glb'
            i += 1
        print(f'Output file: {output_path}')

    with open(input_path, 'rb') as f:
        iqm = f.read()

    options = {'model_name': os.path.basename(input_path)}
    glb = iqm2glb(iqm, options)

    with open(output_path, 'wb') as f:
        glb.write(f)
