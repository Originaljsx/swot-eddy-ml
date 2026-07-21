"""From Satellite Strip to Frontal Eddies — SWOT -> PET -> ADT pipeline explainer."""
from manim import *

BG = "#111418"
CYC = "#58a6ff"     # cyclonic / cold
ANT = "#f85149"     # anticyclonic / warm
GOLD = "#e3b341"    # front line / frontal eddies
MUTE = "#8b949e"    # NaN / inpainted / untrusted
config.background_color = BG

A = "assets/"


def swath_img(path, height=6.0):
    im = ImageMobject(A + path)
    im.set(height=height)
    return im


## Scene1_Hook
class Scene1_Hook(Scene):
    def construct(self):
        ## Scene1_Hook.title
        title = Text("Finding Frontal Eddies in SWOT Altimetry", font_size=40, color=WHITE)
        sub = Text("crop → tile → inpaint → contours → front distance",
                   font_size=24, color=MUTE).next_to(title, DOWN)
        self.add_subcaption("Finding frontal eddies in SWOT altimetry.", duration=2.5)
        self.play(Write(title), FadeIn(sub, shift=UP * 0.3))
        self.wait(1)
        self.play(FadeOut(title), FadeOut(sub))

        ## Scene1_Hook.swath
        adt = swath_img("adt_full.png", 6.8).shift(LEFT * 3.5)
        cap = VGroup(
            Text("One SWOT pass over the Gulf Stream", font_size=30, color=WHITE),
            Text("absolute dynamic topography (ADT)", font_size=22, color=MUTE),
        ).arrange(DOWN, aligned_edge=LEFT).shift(RIGHT * 2.5)
        self.add_subcaption("This is one pass of the SWOT satellite over the Gulf Stream.",
                            duration=3)
        self.play(FadeIn(adt, shift=RIGHT), FadeIn(cap))
        self.wait(1.5)

        ## Scene1_Hook.reveal
        overlay = swath_img("front_overlay.png", 6.8).move_to(adt)
        q = Text("Where are the frontal eddies?", font_size=30, color=GOLD).move_to(cap)
        self.add_subcaption("Somewhere in here are frontal eddies — the gold rings. "
                            "This is how they were found.", duration=4)
        self.play(FadeOut(cap), FadeIn(q))
        self.wait(0.5)
        self.play(FadeIn(overlay))
        self.wait(2)
        self.play(*[FadeOut(m) for m in (adt, overlay, q)])


## Scene2_Swath
class Scene2_Swath(Scene):
    def construct(self):
        ## Scene2_Swath.grid
        head = Text("Anatomy of a swath", font_size=34, color=WHITE).to_edge(UP)
        self.play(Write(head))
        n = 23
        squares = VGroup(*[Square(0.28, stroke_width=1, stroke_color=WHITE,
                                  fill_opacity=0.8) for _ in range(n)])
        squares.arrange(RIGHT, buff=0.03).shift(UP * 1.2)
        for i, s in enumerate(squares):
            if 10 <= i <= 12:
                s.set_fill(MUTE).set_stroke(MUTE)
            else:
                t = abs(i - n / 2) / (n / 2)
                s.set_fill(interpolate_color(ManimColor(CYC), ManimColor(ANT), t))
        self.add_subcaption("Across the track: 69 pixels, two kilometres each — "
                            "with a blind strip at nadir, directly under the satellite.",
                            duration=5)
        self.play(LaggedStartMap(FadeIn, squares, lag_ratio=0.04))
        b1 = Brace(squares, DOWN)
        t1 = b1.get_text("69 pixels × 2 km").set_color(WHITE).scale(0.8)
        nadir = Text("nadir gap", font_size=20, color=MUTE).next_to(squares[11], UP)
        self.play(GrowFromCenter(b1), FadeIn(t1), FadeIn(nadir))
        self.wait(1.5)

        ## Scene2_Swath.channels
        ch = VGroup(
            MathTex(r"\text{SSHA} = \texttt{ssha\_filtered}",
                    tex_to_color_map={r"\text{SSHA}": CYC}).scale(0.9),
            Text("sea-level anomaly — eddies are detected on this", font_size=22,
                 color=MUTE),
            MathTex(r"\text{ADT} = \texttt{ssha\_unedited} + \text{MDT}",
                    tex_to_color_map={r"\text{ADT}": GOLD}).scale(0.9),
            Text("absolute height — carries the Gulf Stream itself", font_size=22,
                 color=MUTE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).shift(DOWN * 1.6)
        self.add_subcaption("Two channels: the filtered anomaly for detection, and "
                            "absolute dynamic topography — anomaly plus the mean surface — "
                            "which carries the Gulf Stream.", duration=6)
        self.play(Write(ch[0]), FadeIn(ch[1]))
        self.wait(1)
        self.play(Write(ch[2]), FadeIn(ch[3]))
        self.wait(2)
        self.play(*[FadeOut(m) for m in (head, squares, b1, t1, nadir, ch)])


## Scene3_PadTile
class Scene3_PadTile(Scene):
    def construct(self):
        ## Scene3_PadTile.pad
        head = Text("Pad and tile", font_size=34, color=WHITE).to_edge(UP)
        self.play(Write(head))
        swath = Rectangle(width=1.5, height=5.4, fill_color=CYC, fill_opacity=0.5,
                          stroke_color=WHITE).shift(LEFT * 4)
        lab69 = Text("69 px", font_size=22, color=WHITE).next_to(swath, DOWN)
        self.add_subcaption("Square tiles are needed, so the 69 pixels are padded to 128.",
                            duration=3.5)
        self.play(FadeIn(swath), FadeIn(lab69))
        wingL = Rectangle(width=0.65, height=5.4, fill_color=MUTE, fill_opacity=0.35,
                          stroke_color=MUTE).next_to(swath, LEFT, buff=0)
        wingR = wingL.copy().next_to(swath, RIGHT, buff=0)
        lab128 = Text("128 px (NaN pad)", font_size=22, color=MUTE).next_to(lab69, DOWN, buff=0.15)
        self.play(GrowFromEdge(wingL, RIGHT), GrowFromEdge(wingR, LEFT), FadeIn(lab128))
        self.wait(1)

        ## Scene3_PadTile.tiles
        padded = VGroup(wingL, swath, wingR)
        win = Square(2.8, stroke_color=GOLD, stroke_width=5).move_to(padded.get_top() + DOWN * 1.4)
        wlab = Text("128 × 128 tile", font_size=22, color=GOLD).next_to(win, RIGHT)
        self.add_subcaption("A 128-by-128 window slides along the track…", duration=3)
        self.play(Create(win), FadeIn(wlab))
        self.wait(0.5)
        ghost = win.copy().set_stroke(GOLD, opacity=0.35)
        self.add(ghost)
        self.add_subcaption("…stepping by 64 pixels, so tiles overlap by half — every eddy "
                            "lands fully inside at least one tile.", duration=5)
        self.play(win.animate.shift(DOWN * 1.4), run_time=1.2)
        overlap = Rectangle(width=2.8, height=1.4, fill_color=GOLD, fill_opacity=0.3,
                            stroke_width=0).move_to(win.get_top() + DOWN * 0.7)
        olab = Text("64-px overlap", font_size=22, color=GOLD).next_to(overlap, RIGHT)
        self.play(FadeIn(overlap), FadeIn(olab), FadeOut(wlab))
        self.wait(2)
        self.play(*[FadeOut(m) for m in (head, padded, lab69, lab128, win, ghost,
                                         overlap, olab)])


## Scene4_Inpaint
class Scene4_Inpaint(Scene):
    def construct(self):
        ## Scene4_Inpaint.problem
        head = Text("The interpolation: biharmonic inpainting", font_size=34,
                    color=WHITE).to_edge(UP)
        self.play(Write(head))
        before = swath_img("tile_before.png", 3.2).shift(LEFT * 3.4 + UP * 0.8)
        blab = Text("real tile — 60% missing", font_size=22, color=MUTE).next_to(before, DOWN)
        self.add_subcaption("Contour detection needs closed curves — but real tiles are "
                            "full of holes.", duration=4)
        self.play(FadeIn(before), FadeIn(blab))
        self.wait(1)

        ## Scene4_Inpaint.equation
        eq = MathTex(r"\nabla^4 u = 0 \quad \text{in } \Omega_{\text{gap}}",
                     tex_to_color_map={r"\nabla^4 u": GOLD}).scale(1.1)
        eq2 = MathTex(
            r"\frac{\partial^4 u}{\partial x^4}"
            r"+2\frac{\partial^4 u}{\partial x^2 \partial y^2}"
            r"+\frac{\partial^4 u}{\partial y^4}=0").scale(0.8).set_color(MUTE)
        bc = MathTex(r"u = \text{data}, \;\; \nabla u \ \text{continuous on } \partial\Omega")\
            .scale(0.8).set_color(WHITE)
        grp = VGroup(eq, eq2, bc).arrange(DOWN, buff=0.35).shift(RIGHT * 2.6 + UP * 0.8)
        self.add_subcaption("Every hole is filled by solving the biharmonic equation — "
                            "del to the fourth of u equals zero inside the gap.", duration=5)
        self.play(Write(eq))
        self.play(FadeIn(eq2))
        self.add_subcaption("matching the data in value AND slope at the hole's edge — "
                            "the shape of a thin elastic plate.", duration=4.5)
        self.play(Write(bc))
        self.wait(1)

        ## Scene4_Inpaint.after
        after = swath_img("tile_after.png", 3.2).move_to(before)
        alab = Text("inpainted — smooth fill", font_size=22, color=GOLD).move_to(blab)
        self.add_subcaption("The result: a seamless field where contours can close. "
                            "But filled pixels are remembered — they are invented data.",
                            duration=5.5)
        self.play(FadeTransform(before, after), FadeTransform(blab, alab))
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in (head, after, alab, grp)])


## Scene5_PET
class Scene5_PET(Scene):
    def construct(self):
        ## Scene5_PET.geostrophy
        head = Text("py-eddy-tracker: closed contours = eddies", font_size=34,
                    color=WHITE).to_edge(UP)
        self.play(Write(head))
        rings = VGroup(*[Ellipse(width=0.8 + 0.55 * k, height=0.55 + 0.4 * k,
                                 stroke_color=CYC, stroke_width=3)
                         for k in range(5)]).shift(LEFT * 4 + UP * 0.6)
        dot = Dot(rings.get_center(), color=CYC)
        self.add_subcaption("On the sea surface, water orbits height extrema — geostrophic "
                            "balance turns height gradients into currents.", duration=5)
        self.play(FadeIn(dot), LaggedStartMap(Create, rings, lag_ratio=0.25))
        geo = MathTex(r"u_g = -\frac{g}{f}\frac{\partial \eta}{\partial y},\qquad "
                      r"v_g = \frac{g}{f}\frac{\partial \eta}{\partial x}").scale(0.9)
        geo.shift(RIGHT * 2.4 + UP * 1.3)
        self.play(Write(geo))
        self.wait(1)

        ## Scene5_PET.criteria
        crit = VGroup(
            Text("closed SLA contour around an extremum", font_size=24, color=WHITE),
            Text("amplitude ≥ 1 cm", font_size=24, color=WHITE),
            Text("speed radius R ≥ 5 km   (4–1000 px area)", font_size=24, color=WHITE),
            Text("shape error ≤ 85 %  (near-elliptical)", font_size=24, color=WHITE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).shift(RIGHT * 2.4 + DOWN * 0.9)
        checks = VGroup(*[Text("✓", font_size=26, color=GOLD).next_to(c, LEFT, buff=0.25)
                          for c in crit])
        self.add_subcaption("A candidate becomes an eddy if its contour closes, stands at "
                            "least a centimetre tall, spans five kilometres, and stays "
                            "near-elliptical.", duration=6.5)
        for c, ck in zip(crit, checks):
            self.play(FadeIn(c, shift=RIGHT * 0.3), FadeIn(ck), run_time=0.6)
        self.wait(1)
        self.play(*[FadeOut(m) for m in (rings, dot, geo, crit, checks)])

        ## Scene5_PET.real
        z1 = swath_img("eddy_zoom_1.png", 3.1).shift(LEFT * 4)
        z2 = swath_img("eddy_zoom_2.png", 3.1)
        z3 = swath_img("eddy_zoom_3.png", 3.1).shift(RIGHT * 4)
        caps = VGroup(
            Text("R = 38 km, 10 cm deep", font_size=20, color=CYC).next_to(z1, DOWN),
            Text("R = 37 km, 6 cm", font_size=20, color=CYC).next_to(z2, DOWN),
            Text("R = 37 km, 18 cm", font_size=20, color=CYC).next_to(z3, DOWN),
        )
        note = Text("real confirmed frontal eddies — SSHA with PET speed contour",
                    font_size=22, color=MUTE).to_edge(DOWN)
        self.add_subcaption("These are real detections from the dataset — hand-confirmed "
                            "frontal eddies, with the contour py-eddy-tracker drew.",
                            duration=6)
        self.play(LaggedStart(FadeIn(z1), FadeIn(z2), FadeIn(z3), lag_ratio=0.3),
                  FadeIn(caps), FadeIn(note))
        self.wait(3)
        self.play(*[FadeOut(m) for m in (head, z1, z2, z3, caps, note)])


## Scene6_Reject
class Scene6_Reject(Scene):
    def construct(self):
        ## Scene6_Reject.dedup
        head = Text("Cleaning up: dedup + trust checks", font_size=34,
                    color=WHITE).to_edge(UP)
        self.play(Write(head))
        t1 = Square(2.4, stroke_color=WHITE).shift(LEFT * 4.6 + UP * 0.9)
        t2 = Square(2.4, stroke_color=WHITE).shift(LEFT * 3.4 + UP * 0.9)
        e1 = Circle(0.45, color=CYC).move_to((t1.get_center() + t2.get_center()) / 2)
        e2 = e1.copy()
        self.add_subcaption("Overlapping tiles find the same eddy twice — duplicates "
                            "within ten kilometres are merged.", duration=4.5)
        self.play(Create(t1), Create(t2), Create(e1), Create(e2))
        self.play(e2.animate.set_stroke(opacity=0.2), run_time=0.8)
        dd = Text("dedup: keep one", font_size=22, color=WHITE)\
            .next_to(VGroup(t1, t2), DOWN)
        self.play(FadeIn(dd))
        self.wait(1)

        ## Scene6_Reject.gap
        gap_e = Circle(0.7, color=CYC).shift(RIGHT * 1.2 + UP * 0.9)
        gap_fill = Sector(radius=0.7, angle=PI * 1.3, fill_color=MUTE, fill_opacity=0.8,
                          stroke_width=0).move_arc_center_to(gap_e.get_center())
        gtxt = MathTex(r"\text{gap frac} > 40\% \Rightarrow \text{reject}",
                       tex_to_color_map={r"\text{reject}": ANT}).scale(0.7)
        gtxt.next_to(gap_e, DOWN)
        x1 = Cross(gap_e, stroke_color=ANT, stroke_width=6)
        self.add_subcaption("If most of a contour stands on inpainted — invented — pixels, "
                            "the detection is rejected.", duration=4.5)
        self.play(Create(gap_e), FadeIn(gap_fill), Write(gtxt))
        self.play(Create(x1))
        self.wait(1)

        ## Scene6_Reject.nadir
        strip = Rectangle(width=0.5, height=2.6, fill_color=MUTE, fill_opacity=0.5,
                          stroke_width=0).shift(RIGHT * 4.9 + UP * 0.9)
        nad_e = Circle(0.55, color=ANT).move_to(strip.get_center())
        ntxt = Text("≥70% area in nadir gap → drop", font_size=22, color=WHITE)\
            .next_to(strip, DOWN)
        x2 = Cross(nad_e, stroke_color=ANT, stroke_width=6)
        self.add_subcaption("Smooth blobs invented inside the nadir strip are dropped too.",
                            duration=3.5)
        self.play(FadeIn(strip), Create(nad_e), FadeIn(ntxt))
        self.play(Create(x2))
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in (head, t1, t2, e1, e2, dd, gap_e, gap_fill,
                                         gtxt, x1, strip, nad_e, ntxt, x2)])


## Scene7_ADTFront
class Scene7_ADTFront(Scene):
    def construct(self):
        ## Scene7_ADTFront.profile
        head = Text("The ADT trick: one contour finds the Gulf Stream", font_size=32,
                    color=WHITE).to_edge(UP)
        self.play(Write(head))
        ax = Axes(x_range=[0, 10, 2], y_range=[0, 1, 0.35],
                  x_length=6, y_length=3,
                  axis_config={"color": MUTE, "include_ticks": False}).shift(DOWN * 0.6)
        xl = Text("distance across swath", font_size=20, color=MUTE)\
            .next_to(ax, DOWN, buff=0.15)
        yl = Text("ADT (m)", font_size=20, color=MUTE).rotate(PI / 2)\
            .next_to(ax, LEFT, buff=0.15)
        curve = ax.plot(lambda x: 0.5 + 0.42 * np.tanh(1.2 * (x - 5)), color=WHITE)
        shelf = Text("shelf (low)", font_size=20, color=CYC)\
            .move_to(ax.c2p(1.6, 0.25))
        sarg = Text("Sargasso (high)", font_size=20, color=ANT)\
            .move_to(ax.c2p(8.2, 0.85))
        self.add_subcaption("Across the Gulf Stream, absolute height is a step: low on the "
                            "shelf side, high on the Sargasso side, with the jet at the jump.",
                            duration=6)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl))
        self.play(Create(curve), FadeIn(shelf), FadeIn(sarg))
        line35 = DashedLine(ax.c2p(0, 0.5), ax.c2p(10, 0.5), color=GOLD)
        l35 = MathTex(r"\text{ADT}=0.35\,\text{m}", color=GOLD).scale(0.7)\
            .next_to(line35, UP, aligned_edge=RIGHT).shift(LEFT * 0.4)
        cross = Dot(ax.c2p(5, 0.5), color=GOLD, radius=0.09)
        self.add_subcaption("So the level-line at 0.35 metres crosses the profile exactly "
                            "at the jet — in every pass.", duration=4.5)
        self.play(Create(line35), Write(l35))
        self.play(FadeIn(cross, scale=2.5))
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in (ax, xl, yl, curve, shelf, sarg, line35, l35, cross)])

        ## Scene7_ADTFront.maps
        grad = swath_img("grad_adt.png", 4.9).shift(LEFT * 4.8 + UP * 0.5)
        glab = Text("|∇ADT| ridge", font_size=20, color=WHITE).next_to(grad, DOWN, buff=0.15)
        ov = swath_img("front_overlay.png", 4.9).shift(LEFT * 2.1 + UP * 0.5)
        olab = Text("front line + eddies", font_size=20, color=GOLD)\
            .next_to(ov, DOWN, buff=0.15)
        self.add_subcaption("In two dimensions the gradient magnitude shows the jet as a "
                            "bright ridge — and the gold contour rides it.", duration=5.5)
        self.play(FadeIn(grad), FadeIn(glab))
        self.play(FadeIn(ov), FadeIn(olab))
        eqs = VGroup(
            MathTex(r"F=\{\,x:\ \text{ADT}(x)=0.35\,\}",
                    tex_to_color_map={"F": GOLD}).scale(0.8),
            Text("open contours only — closed rings are eddies", font_size=20, color=MUTE),
            MathTex(r"d(e)=\operatorname{sign}\!\big(\text{ADT}(x_e)-0.35\big)\cdot"
                    r"\min_{p\in F}\lVert x_e-p\rVert").scale(0.68),
            Text("− inshore side   |   + offshore side", font_size=20, color=MUTE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).shift(RIGHT * 3.1 + UP * 1.4)
        self.add_subcaption("Each eddy gets a signed distance to this line: negative on the "
                            "coastal side, positive offshore.", duration=5)
        self.play(Write(eqs[0]), FadeIn(eqs[1]))
        self.play(Write(eqs[2]), FadeIn(eqs[3]))
        self.wait(1.5)

        ## Scene7_ADTFront.separation
        med_f = VGroup(Rectangle(width=0.4, height=0.35, fill_color=GOLD,
                                 fill_opacity=0.9, stroke_width=0),
                       Text("frontal: median 22 km", font_size=20, color=GOLD))\
            .arrange(RIGHT, buff=0.2)
        med_o = VGroup(Rectangle(width=2.6, height=0.35, fill_color=CYC,
                                 fill_opacity=0.7, stroke_width=0),
                       Text("open ocean: 216 km", font_size=20, color=CYC))\
            .arrange(RIGHT, buff=0.2)
        bars = VGroup(med_f, med_o).arrange(DOWN, aligned_edge=LEFT, buff=0.25)\
            .shift(RIGHT * 2.6 + DOWN * 1.2)
        auc = Text("AUC 0.95 — distance alone nearly separates them",
                   font_size=22, color=WHITE).next_to(bars, DOWN, buff=0.35)\
            .align_to(bars, LEFT).shift(LEFT * 0.5)
        self.add_subcaption("Validated on over 800 hand labels: frontal eddies sit a median "
                            "of 22 kilometres from the line; open-ocean eddies over 200. "
                            "AUC 0.95.", duration=6.5)
        self.play(GrowFromEdge(med_f[0], LEFT), FadeIn(med_f[1]))
        self.play(GrowFromEdge(med_o[0], LEFT), FadeIn(med_o[1]))
        self.play(Write(auc))
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in (head, grad, glab, ov, olab, eqs, med_f,
                                         med_o, auc)])


## Scene8_Pipeline
class Scene8_Pipeline(Scene):
    def construct(self):
        ## Scene8_Pipeline.chain
        head = Text("The whole pipeline", font_size=34, color=WHITE).to_edge(UP)
        self.play(Write(head))
        steps = VGroup(
            Text("SWOT pass", font_size=24, color=WHITE),
            Text("crop · pad · tile", font_size=24, color=WHITE),
            MathTex(r"\nabla^4 u = 0", color=GOLD).scale(0.8),
            Text("PET contours", font_size=24, color=CYC),
            Text("ADT front distance", font_size=24, color=GOLD),
            Text("frontal candidates", font_size=24, color=GOLD),
        ).arrange(DOWN, buff=0.42).shift(LEFT * 3.8 + DOWN * 0.4)
        arrows = VGroup(*[Arrow(steps[i].get_bottom(), steps[i + 1].get_top(),
                                buff=0.06, stroke_width=3, color=MUTE,
                                max_tip_length_to_length_ratio=0.4)
                          for i in range(len(steps) - 1)])
        self.add_subcaption("From raw strip to frontal-eddy candidates: crop, pad, tile; "
                            "fill the gaps with one PDE; close the contours; measure the "
                            "distance to one gold line.", duration=8)
        for s, a in zip(steps, [*arrows, None]):
            self.play(FadeIn(s, shift=UP * 0.2), run_time=0.5)
            if a is not None:
                self.play(GrowArrow(a), run_time=0.3)

        ## Scene8_Pipeline.payoff
        ov = swath_img("front_overlay.png", 5.4).shift(RIGHT * 2.2)
        met = VGroup(
            Text("84% precision / 72% recall", font_size=26, color=WHITE),
            Text("among real eddies (n = 802 labels)", font_size=20, color=MUTE),
        ).arrange(DOWN).next_to(ov, RIGHT, buff=0.4).shift(LEFT * 0.2)
        self.add_subcaption("Two contour tricks and one PDE: eighty-four percent precision, "
                            "seventy-two percent recall among real eddies.", duration=6)
        self.play(FadeIn(ov))
        self.play(FadeIn(met))
        self.wait(3)
        self.play(*[FadeOut(m) for m in (head, steps, arrows, ov, met)])
        fin = Text("swot-eddy-ml", font_size=30, color=MUTE)
        self.play(FadeIn(fin))
        self.wait(1.5)
        self.play(FadeOut(fin))
