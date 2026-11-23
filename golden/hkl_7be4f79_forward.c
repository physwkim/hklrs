/* Golden forward vectors for the geometries only hkl 7be4f79 has: PYFAI
 * (registered) and PETRA3 P08 LISA (its REGISTER_DIFFRACTOMETER is commented
 * out, so the engine file is #included and its static geometry used).
 * Same schema as hkl_forward.json. Build and run: golden/hkl_7be4f79_forward.sh */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "hkl.h"
#include "hkl/hkl-geometry-private.h"
#include "hkl/hkl-sample-private.h"
#include "hkl/hkl-matrix-private.h"
#include "hkl/hkl-engine-petra3-p08-lisa.c"

static uint64_t rng = 20260101;
static double uni(double lo, double hi)
{
	uint64_t z = (rng += 0x9e3779b97f4a7c15ULL);
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
	z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
	z ^= z >> 31;
	return lo + (hi - lo) * ((z >> 11) * (1.0 / 9007199254740992.0));
}

typedef struct { const char *name; double a, b, c, al, be, ga, ux, uy, uz; } Cell;
static const Cell cells[] = {
	{"cubic", 1.54, 1.54, 1.54, 90, 90, 90, 0, 0, 0},
	{"triclinic", 4.1, 5.3, 6.7, 82, 97, 101, 12, -31, 47},
};
static const double wls[] = {1.54, 0.71};

/* user unit -> (is translation, canonical deg|mm -> user factor) */
static int unit_kind(const char *u, double *f, int *trans)
{
	*trans = 0; *f = 1;
	if (!strcmp(u, "Degree")) return 1;
	if (!strcmp(u, "Milli Radian")) { *f = M_PI / 180 * 1000; return 1; }
	if (!strcmp(u, "Milli Meter")) { *trans = 1; return 1; }
	if (!strcmp(u, "Meter")) { *trans = 1; *f = 1e-3; return 1; }
	return 0;
}

static void pm(const HklMatrix *m)
{
	printf("[");
	for (int i = 0; i < 3; i++) {
		printf("%s[", i ? ", " : "");
		for (int j = 0; j < 3; j++) printf("%s%.17g", j ? ", " : "", hkl_matrix_get(m, i, j));
		printf("]");
	}
	printf("]");
}

typedef struct { const char *name; int lisa; } Geo;

/* a geometry built without a factory has no axis-name list (it is read from
 * the factory), so LISA names its axes itself */
static size_t axis_names(const Geo *g, HklGeometry *geo, const char *const **out)
{
	if (g->lisa) { *out = hkl_geometry_petra3_p08_lisa_axes; return 5; }
	const darray_string *d = hkl_geometry_axis_names_get(geo);
	*out = (const char *const *)d->item;
	return darray_size(*d);
}

static HklGeometry *new_geo(const Geo *g, const HklFactory **fac)
{
	/* hkl_geometry_new ignores its ops for a NULL factory (readonly geometry),
	 * so LISA, unregistered upstream, gets a stand-in factory */
	static const HklFactory lisa = {.name = "PETRA3 P08 LISA", .description = ""};
	if (g->lisa) return hkl_geometry_new_petra3_p08_lisa(&lisa);
	*fac = hkl_factory_get_by_name(g->name, NULL);
	return hkl_factory_create_new_geometry(*fac);
}

int main(void)
{
	static const Geo geos[] = {{"PYFAI", 0}, {"PETRA3 P08 LISA", 1}};
	int first_case = 1;
	printf("{\"hkl_version\": \"7be4f79\", \"seed\": %llu,\n \"units\": {\"angle\": \"deg\", \"length\": \"mm\", \"wavelength\": \"angstrom\"},\n \"geometries\": {", (unsigned long long)rng);
	for (int gi = 0; gi < 2; gi++) {
		const HklFactory *fac = NULL;
		HklGeometry *g = new_geo(&geos[gi], &fac);
		const char *const *names;
		size_t nn = axis_names(&geos[gi], g, &names);
		printf("%s\"%s\": {\"axes\": [", gi ? ", " : "", geos[gi].name);
		for (size_t i = 0; i < nn; i++) {
			const char *n = names[i];
			const HklParameter *p = hkl_geometry_axis_get(g, n, NULL);
			const HklVector *v = hkl_parameter_axis_v_get(p);
			printf("%s{\"name\": \"%s\", \"axis_v\": ", i ? ", " : "", n);
			if (v) printf("[%.17g, %.17g, %.17g]", v->data[0], v->data[1], v->data[2]); else printf("null");
			printf(", \"unit\": \"%s\"}", hkl_parameter_user_unit_get(p));
		}
		printf("]}");
		hkl_geometry_free(g);
	}
	printf("},\n \"cases\": [\n");
	for (int gi = 0; gi < 2; gi++)
	for (size_t ci = 0; ci < 2; ci++)
	for (size_t wi = 0; wi < 2; wi++)
	for (int k = 0; k < 10; k++) {
		const Geo *G = &geos[gi]; const Cell *C = &cells[ci];
		const HklFactory *fac = NULL;
		HklGeometry *g = new_geo(G, &fac);
		const char *const *names;
		size_t n = axis_names(G, g, &names);
		double canon[16], vals[16];
		for (size_t i = 0; i < n; i++) {
			double f; int t;
			unit_kind(hkl_parameter_user_unit_get(hkl_geometry_axis_get(g, names[i], NULL)), &f, &t);
			canon[i] = t ? uni(-50, 50) : uni(-180, 180);
			vals[i] = canon[i] * f;
		}
		GError *e = NULL;
		int ok = hkl_geometry_wavelength_set(g, wls[wi], HKL_UNIT_USER, &e);
		ok = ok && hkl_geometry_axis_values_set(g, vals, n, HKL_UNIT_USER, &e);
		HklSample *s = hkl_sample_new("golden");
		HklLattice *l = hkl_lattice_new(C->a, C->b, C->c, C->al * HKL_DEGTORAD, C->be * HKL_DEGTORAD, C->ga * HKL_DEGTORAD, &e);
		hkl_sample_lattice_set(s, l);
		HklMatrix *U = hkl_matrix_new_euler(C->ux * HKL_DEGTORAD, C->uy * HKL_DEGTORAD, C->uz * HKL_DEGTORAD);
		hkl_sample_U_set(s, U, NULL);
		double h[3] = {0};
		const char *err = NULL;
		if (!ok) err = "set failed";
		else if (G->lisa) {
			/* body of hkl_mode_hkl_petra3_p08_lisa_get_real with the real ops */
			HklDetector *d = hkl_detector_factory_new(HKL_DETECTOR_TYPE_0D);
			HklMatrix RUB; HklVector hkl, ki, Q;
			hkl_geometry_update(g);
			HklHolder *sh = hkl_geometry_sample_holder_get(g, s);
			hkl_quaternion_to_matrix(&sh->q, &RUB);
			hkl_matrix_times_matrix(&RUB, hkl_sample_UB_get(s));
			ki = hkl_geometry_ki_get(g);
			Q = hkl_geometry_kf_get(g, d);
			hkl_vector_minus_vector(&Q, &ki);
			hkl_matrix_solve(&RUB, &hkl, &Q);
			memcpy(h, hkl.data, sizeof h);
		} else {
			HklDetector *d = hkl_detector_factory_new(HKL_DETECTOR_TYPE_0D);
			HklEngineList *el = hkl_factory_create_new_engine_list(fac);
			hkl_engine_list_init(el, g, d, s);
			HklEngine *en = hkl_engine_list_engine_get_by_name(el, "hkl", NULL);
			if (!en) err = "no 'hkl' engine";
			else {
				hkl_engine_list_get(el);
				if (!hkl_engine_pseudo_axis_values_get(en, h, 3, HKL_UNIT_USER, NULL)) err = "pseudo get failed";
			}
		}
		printf("%s  {\"geometry\": \"%s\", \"sample\": \"%s\", \"lattice\": {\"a\": %.17g, \"b\": %.17g, \"c\": %.17g, \"alpha\": %.17g, \"beta\": %.17g, \"gamma\": %.17g, \"ux\": %.17g, \"uy\": %.17g, \"uz\": %.17g}, \"wavelength\": %.17g, \"axes\": {",
		       first_case ? "" : ",\n", G->name, C->name, C->a, C->b, C->c, C->al, C->be, C->ga, C->ux, C->uy, C->uz, wls[wi]);
		first_case = 0;
		for (size_t i = 0; i < n; i++) printf("%s\"%s\": %.17g", i ? ", " : "", names[i], vals[i]);
		printf("}, \"axes_canonical_deg_mm\": {");
		for (size_t i = 0; i < n; i++) printf("%s\"%s\": %.17g", i ? ", " : "", names[i], canon[i]);
		printf("}, \"index\": %d, \"U\": ", k);
		pm(hkl_sample_U_get(s));
		printf(", \"UB\": "); pm(hkl_sample_UB_get(s));
		if (err) printf(", \"error\": \"%s\"", err);
		else printf(", \"pseudo_names\": [\"h\", \"k\", \"l\"], \"hkl\": [%.17g, %.17g, %.17g]", h[0], h[1], h[2]);
		printf("}");
	}
	printf("\n]}\n");
	return 0;
}
