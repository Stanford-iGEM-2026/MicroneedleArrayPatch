import adsk.core, adsk.fusion, adsk.cam, traceback, math
 
 
def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        design = adsk.fusion.Design.cast(app.activeProduct)
        if not design:
            ui.messageBox('No active Fusion design. Create/open a design first.')
            return
 
        rootComp = design.rootComponent
        tempMgr = adsk.fusion.TemporaryBRepManager.get()
 
        # =========================================================
        # UNIT HELPERS  (spec is given in mm; Fusion API is always cm)
        # =========================================================
        def mm(v):
            return v * 0.1
 
        def P(x, y, z):
            """Point3D from mm coordinates."""
            return adsk.core.Point3D.create(mm(x), mm(y), mm(z))
 
        # =========================================================
        # PARAMETERS (all in mm unless noted)  -- requirement # in comments
        # =========================================================
        # --- Req 3: patch footprint & overall height ---
        BLOCK_X = 8.0
        BLOCK_Y = 5.0
        NEEDLE_HEIGHT = 0.9                       # Req 4
        TOTAL_HEIGHT = 6.04                       # Req 3
        BLOCK_Z = TOTAL_HEIGHT - NEEDLE_HEIGHT     # = 5.14mm base slab height
 
        # --- Req 1/2: grid spacing (achieved via the branch-tree math below,
        # not hardcoded here -- see LEVEL_* parameters) ---
        # column pitch target: 2mm | row pitch target: 1mm | row indent: 1mm
 
        # --- Req 4: biconic needle profile ---
        # NOTE: base/mid diameters were not given explicitly in the spec;
        # these are reasonable assumed defaults, safely smaller than the
        # ~1.41mm minimum needle-to-needle spacing in the grid. Adjust freely.
        NEEDLE_BASE_RADIUS = 0.15    # mm (0.30mm base diameter)
        NEEDLE_MID_RADIUS  = 0.075   # mm (0.15mm diameter at the 0.7mm break)
        NEEDLE_TIP_RADIUS  = 0.002   # mm (near-zero, avoids a degenerate cone)
        NEEDLE_SHALLOW_HEIGHT = 0.7  # mm -- less-steep lower section
        NEEDLE_STEEP_HEIGHT   = 0.2  # mm -- steep section closest to the tip
        FILLET_RADIUS = 0.05         # mm -- smooths needle base into top face
 
        # --- Req 5/6: branching delivery channel ---
        SYRINGE_HOLE_DIA = 0.8       # mm -- Req 5 (fixed, this is the inlet diameter)
        CHANNEL_INLET_RADIUS = SYRINGE_HOLE_DIA / 2.0
 
        # Channel narrows in two stages so there's clearly MORE solid volume
        # than channel volume:
        #   1) TRUNK_RATIO: a sharp step-down right after leaving the 0.8mm
        #      inlet (like a nozzle constriction) -- makes the branches
        #      visibly thinner immediately, not just gradually.
        #   2) A gentle per-level taper through the 4 branch levels, solved
        #      so the channel is already close to TARGET_LEAF_RADIUS by the
        #      time it reaches each needle -- i.e. it keeps getting thinner
        #      the closer it gets, leaving only a small final taper down to
        #      the fixed 0.05mm (0.1mm dia) outlet hole.
        CHANNEL_TRUNK_RATIO   = 0.30   # trunk end radius = inlet radius * this
        TARGET_LEAF_RADIUS_MM = 0.06   # channel radius just before each needle's riser
        trunk_radius = CHANNEL_INLET_RADIUS * CHANNEL_TRUNK_RATIO   # 0.4 * 0.3 = 0.12mm
 
        # Branch angles exactly as specified (Req 6), each level given as the
        # "positive" azimuth -- its sibling is automatically azimuth+180.
        LEVEL_AZIMUTHS_DEG = [135.0, 90.0, 0.0, 0.0]
 
        # Segment length PER LEVEL, solved so the 16 leaf endpoints land
        # exactly on the required 2mm/1mm staggered grid (derivation in the
        # chat write-up): every leaf traverses the same 4 lengths (only the
        # +/- signs differ), so path length to every needle is identical,
        # satisfying "same distance to reach each needle."
        LEVEL_LENGTHS_MM = [0.5 * math.sqrt(2.0), 1.0, 2.0, 1.0]
 
        # Solve the per-level radius ratio so 4 successive shrinks take the
        # channel from trunk_radius down to TARGET_LEAF_RADIUS_MM exactly.
        CHANNEL_LEVEL_RATIO = (TARGET_LEAF_RADIUS_MM / trunk_radius) ** (1.0 / len(LEVEL_AZIMUTHS_DEG))
 
        # --- Req 7: outlet hole ---
        HOLE_DEPTH_BELOW_TIP = 0.2   # mm -- also exactly the shallow/steep
                                      # break height (0.9 - 0.2 = 0.7mm), so
                                      # the outlet sits right at that shoulder
        HOLE_RADIUS = 0.05           # mm (0.1mm diameter)
 
        # Vertical budget: trunk + 4 levels share (BLOCK_Z - SAFETY_MARGIN),
        # NOT the full BLOCK_Z. The last branch level approaches at a shallow
        # angle (not straight up) and has real thickness, so if it were
        # allowed to land exactly ON the top surface, its tilted, angled
        # sweep pokes slightly ABOVE the surface on the "uphill" side --
        # carving an asymmetric notch out of the needle base when subtracted
        # (this is the gap seen at the needle base). Keeping the branch
        # network's last joint safely embedded below the surface, and only
        # closing the remaining distance with a straight VERTICAL riser
        # (added below, per needle), keeps the base surface fully closed.
        SAFETY_MARGIN = 0.2  # mm -- comfortably larger than the channel radii involved
        Z_STEP = (BLOCK_Z - SAFETY_MARGIN) / 5.0
        TRUNK_RISE = Z_STEP
 
        # Every joint sphere is given a radius slightly LARGER than the pipes
        # meeting there. Without this, a sphere sized to exactly match its
        # cylinders only touches them along a single tangent circle (a
        # near-coincident face), which is exactly what triggers the
        # ASM_INCONS_FACE boolean error. A modest oversize guarantees real
        # volumetric overlap at every junction.
        OVERLAP_FACTOR = 1.15
 
        solidBodies = []    # block + needles -> union
        channelBodies = []  # inlet + branches + risers + holes -> union, then subtracted
 
        # =========================================================
        # PRIMITIVE HELPERS
        # =========================================================
        def add_cyl(p1, r1, p2, r2, target):
            body = tempMgr.createCylinderOrCone(P(*p1), mm(r1), P(*p2), mm(r2))
            target.append(body)
 
        def add_sph(c, r, target):
            body = tempMgr.createSphere(P(*c), mm(r))
            target.append(body)
 
        # =========================================================
        # Req 3: BASE BLOCK (rectangular prism)
        # =========================================================
        obb = adsk.core.OrientedBoundingBox3D.create(
            P(0, 0, BLOCK_Z / 2.0),
            adsk.core.Vector3D.create(1, 0, 0),
            adsk.core.Vector3D.create(0, 1, 0),
            mm(BLOCK_X), mm(BLOCK_Y), mm(BLOCK_Z))
        solidBodies.append(tempMgr.createBox(obb))
 
        # =========================================================
        # Req 5/6: BRANCHING CHANNEL -- recursive builder.
        # Collects each leaf's (x, y, radius) so needles are placed at
        # EXACTLY the same coordinates the channel actually reaches.
        # =========================================================
        leaves = []  # list of (x_mm, y_mm, z_mm, radius_mm) at the top of the block
 
        def build_channel(x0, y0, z0, radius0, level):
            if level == len(LEVEL_AZIMUTHS_DEG):
                leaves.append((x0, y0, z0, radius0))
                return
            az = LEVEL_AZIMUTHS_DEG[level]
            L = LEVEL_LENGTHS_MM[level]
            r1 = radius0 * CHANNEL_LEVEL_RATIO
            for az_this in (az, az + 180.0):
                theta = math.radians(az_this)
                x1 = x0 + L * math.cos(theta)
                y1 = y0 + L * math.sin(theta)
                z1 = z0 + Z_STEP
                add_cyl((x0, y0, z0), radius0, (x1, y1, z1), r1, channelBodies)
                # Req 8: sphere joint at every branch point -> guarantees the
                # channel network is a single connected solid, never pinched
                # or disconnected at a split.
                add_sph((x1, y1, z1), r1 * OVERLAP_FACTOR, channelBodies)
                build_channel(x1, y1, z1, r1, level + 1)
 
        # Trunk: straight vertical run from the syringe inlet up to the
        # first branch point. Extends slightly below z=0 so the boolean
        # subtract cleanly punches all the way through the bottom face.
        # This is where the sharp post-inlet narrowing happens (0.4mm -> 0.12mm).
        add_cyl((0, 0, -0.05), CHANNEL_INLET_RADIUS, (0, 0, TRUNK_RISE), trunk_radius, channelBodies)
        add_sph((0, 0, TRUNK_RISE), trunk_radius * OVERLAP_FACTOR, channelBodies)
 
        build_channel(0.0, 0.0, TRUNK_RISE, trunk_radius, 0)
 
        if len(leaves) != 16:
            ui.messageBox(f'Warning: expected 16 needle leaves, got {len(leaves)}.')
 
        # =========================================================
        # Req 4: NEEDLES + Req 7: riser & lateral outlet hole per needle
        # =========================================================
        for (lx, ly, lz, lr) in leaves:
            base_z = BLOCK_Z
            mid_z = BLOCK_Z + NEEDLE_SHALLOW_HEIGHT
            tip_z = BLOCK_Z + NEEDLE_HEIGHT
 
            # Biconic needle: shallow taper (0 -> 0.7mm), then steep taper
            # (0.7mm -> 0.9mm tip). Base fillet is applied later as a real
            # Fusion fillet feature (needs a committed body/edge).
            add_cyl((lx, ly, base_z), NEEDLE_BASE_RADIUS,
                    (lx, ly, mid_z), NEEDLE_MID_RADIUS, solidBodies)
            add_cyl((lx, ly, mid_z), NEEDLE_MID_RADIUS,
                    (lx, ly, tip_z), NEEDLE_TIP_RADIUS, solidBodies)
            # Small weld sphere at the shallow/steep shoulder -- both cones
            # meet there at the exact same radius (a tangent, near-coincident
            # face), the same class of issue that causes ASM_INCONS_FACE.
            add_sph((lx, ly, mid_z), NEEDLE_MID_RADIUS * OVERLAP_FACTOR, solidBodies)
 
            # Req 7: riser from the last branch joint straight up (xy already
            # matches the needle center exactly) to 0.2mm below the tip --
            # i.e. exactly at mid_z, the shallow/steep shoulder.
            hole_z = base_z + (NEEDLE_HEIGHT - HOLE_DEPTH_BELOW_TIP)
            add_cyl((lx, ly, lz), lr, (lx, ly, hole_z), HOLE_RADIUS, channelBodies)
            add_sph((lx, ly, hole_z), HOLE_RADIUS * OVERLAP_FACTOR, channelBodies)
 
            # Lateral bore: breaks through the needle side wall at hole_z so
            # the channel actually opens to the surface (a microneedle can't
            # deliver fluid through a blind internal cavity). Bored generously
            # past the wall to guarantee a clean through-cut.
            bore_len = NEEDLE_MID_RADIUS * 1.6
            add_cyl((lx, ly, hole_z), HOLE_RADIUS,
                    (lx + bore_len, ly, hole_z), HOLE_RADIUS, channelBodies)
 
        # =========================================================
        # UNION solid pieces; UNION channel pieces; SUBTRACT channel
        # from solid so the delivery network is a real hollow cavity.
        # =========================================================
        solidMerged = solidBodies[0]
        for b in solidBodies[1:]:
            tempMgr.booleanOperation(solidMerged, b, adsk.fusion.BooleanTypes.UnionBooleanType)
 
        channelMerged = channelBodies[0]
        for b in channelBodies[1:]:
            tempMgr.booleanOperation(channelMerged, b, adsk.fusion.BooleanTypes.UnionBooleanType)
 
        tempMgr.booleanOperation(solidMerged, channelMerged, adsk.fusion.BooleanTypes.DifferenceBooleanType)
 
        # =========================================================
        # COMMIT TO THE DESIGN (needs an active base feature)
        # =========================================================
        baseFeature = rootComp.features.baseFeatures.add()
        baseFeature.startEdit()
        patchBody = rootComp.bRepBodies.add(solidMerged, baseFeature)
        patchBody.name = "MicroneedlePatch"
        baseFeature.finishEdit()
 
        # =========================================================
        # Req 4: FILLET the 16 needle-base edges (where each needle
        # meets the flat top surface) for a smooth, printable transition.
        # =========================================================
        try:
            edgeCol = adsk.core.ObjectCollection.create()
            target_z = mm(BLOCK_Z)
            target_r = mm(NEEDLE_BASE_RADIUS)
            for edge in patchBody.edges:
                geom = edge.geometry
                if geom.curveType == adsk.core.Curve3DTypes.Circle3DCurveType:
                    circle = adsk.fusion.Circle3D.cast(geom)
                    if circle and abs(circle.center.z - target_z) < mm(0.01) \
                            and abs(circle.radius - target_r) < mm(0.02):
                        edgeCol.add(edge)
 
            if edgeCol.count > 0:
                filletFeats = rootComp.features.filletFeatures
                filletInput = filletFeats.createInput()
                radiusVal = adsk.core.ValueInput.createByReal(mm(FILLET_RADIUS))
                filletInput.edgeSetInputs.addConstantRadiusEdgeSet(edgeCol, radiusVal, True)
                filletFeats.add(filletInput)
            else:
                ui.messageBox('Note: could not auto-detect needle-base edges for filleting; '
                               'you can select them manually and apply a ~0.05mm fillet.')
        except:
            ui.messageBox('Note: automatic fillet step failed; the patch body was still created '
                           'successfully. You can add the base fillets manually.\n' + traceback.format_exc())
 
        ui.messageBox(
            "Microneedle patch created:\n"
            f" - Patch: {BLOCK_X} x {BLOCK_Y} x {TOTAL_HEIGHT} mm\n"
            f" - {len(leaves)} needles, 4 rows x 4 cols, 2mm/1mm staggered grid\n"
            f" - Needle: base R{NEEDLE_BASE_RADIUS}mm, biconic 0.7mm+0.2mm, H{NEEDLE_HEIGHT}mm\n"
            f" - Syringe inlet: {SYRINGE_HOLE_DIA}mm diameter\n"
            f" - Outlet holes: {HOLE_RADIUS*2}mm diameter, {HOLE_DEPTH_BELOW_TIP}mm below each tip")
 
    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))