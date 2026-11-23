/* binoculars-ng 7be4f79's HDF5 cube I/O on the merged cube of
 * bin_7be4f79_cube.c, hkl_binoculars_cube_new_merge of the two half scans.
 *   save FILE     writes it with hkl_binoculars_cube_save_hdf5 and CONFIG
 *   load FILE...  reads each FILE with hkl_binoculars_cube_new_from_file and
 *                 prints, per file, whether its axes (name, index, resolution,
 *                 imin, imax), offset0, photons and contributions equal the
 *                 merged cube's; cube_cmp is binoculars' own comparison,
 *                 which skips the bins
 * widening them (BIN-6), so the merged cube's offset0 is printed next to
 * the one its axes give (compute_offset0, hkl-binoculars.c:1654) and the
 * loads are compared against the latter.
 * Build and run: bin_7be4f79_hdf5.sh */
#include <string.h>
#include "bin_7be4f79_scan.h"

static const char config[] = "[input]\ntype = bin_7be4f79_scan.h\n"
	"[projection]\ntype = qcustom qx_qy_qz\nresolution = 0.0214, 0.0177, 0.0199\n";

static HklBinocularsCube *merged(void)
{
	double pix[3 * N];
	uint8_t mask[N];
	HklGeometry *g = scan_new(pix, mask);
	HklBinocularsSpace *spaces[FRAMES];
	HklBinocularsCube *halves[2] = {hkl_binoculars_cube_new_empty(),
					hkl_binoculars_cube_new_empty()};
	int saved = quiet();
	for (int f = 0; f < FRAMES; f++) {
		scan_frame(g, f);
		spaces[f] = hkl_binoculars_space_new(N, 3);
		space(spaces[f], g, f, pix, resolutions, mask);
		hkl_binoculars_cube_add_space(halves[f >= FRAMES / 2], spaces[f]);
	}
	HklBinocularsCube *m = hkl_binoculars_cube_new_merge(halves[0], halves[1]);
	loud(saved);
	return m;
}

static ptrdiff_t offset0_from_axes(const HklBinocularsCube *c)
{
	ptrdiff_t len = 1, offset = 0;
	for (size_t i = darray_size(c->axes); i-- > 0;) {
		const HklBinocularsAxis *a = &darray_item(c->axes, i);
		offset += len * a->imin;
		len *= a->imax - a->imin + 1;
	}
	return offset;
}

static void json_string(const char *s)
{
	putchar('"');
	for (; *s; s++)
		if (*s == '\n') printf("\\n");
		else if (*s == '"' || *s == '\\') printf("\\%c", *s);
		else putchar(*s);
	putchar('"');
}

static const char *flag(int v) { return v ? "true" : "false"; }

static void verdict(const char *fn, const HklBinocularsCube *want, int first)
{
	const char *base = strrchr(fn, '/') ? strrchr(fn, '/') + 1 : fn;
	int saved = quiet();
	HklBinocularsCube *got = hkl_binoculars_cube_new_from_file(fn);
	unsigned int cmp = got ? hkl_binoculars_cube_cmp(got, want) : 1;
	loud(saved);
	printf("%s  \"%s\": ", first ? "" : ",\n", base);
	if (!got) {
		printf("{\"loaded\": false}");
		return;
	}
	int axes = darray_size(got->axes) == darray_size(want->axes);
	size_t n = 1;
	for (size_t i = 0; axes && i < darray_size(want->axes); i++) {
		const HklBinocularsAxis *a = &darray_item(got->axes, i), *b = &darray_item(want->axes, i);
		axes &= a->name == b->name && a->index == b->index && a->resolution == b->resolution
			&& a->imin == b->imin && a->imax == b->imax;
		n *= b->imax - b->imin + 1;
	}
	printf("{\"loaded\": true, \"axes\": %s, \"offset0\": %s, \"photons\": %s, "
	       "\"contributions\": %s, \"cube_cmp\": %u}",
	       flag(axes), flag(got->offset0 == want->offset0),
	       flag(axes && !memcmp(got->photons, want->photons, n * sizeof(*want->photons))),
	       flag(axes && !memcmp(got->contributions, want->contributions,
				    n * sizeof(*want->contributions))),
	       cmp);
}

int main(int argc, char **argv)
{
	if (argc == 3 && !strcmp(argv[1], "save"))
		return hkl_binoculars_cube_save_hdf5(argv[2], config, merged()) < 0;
	if (argc < 3 || strcmp(argv[1], "load")) {
		fprintf(stderr, "usage: %s save FILE | load FILE...\n", argv[0]);
		return 2;
	}
	HklBinocularsCube *want = merged();
	ptrdiff_t stored = want->offset0;
	want->offset0 = offset0_from_axes(want);
	printf("{\"binoculars_version\": \"7be4f79\", \"config\": ");
	json_string(config);
	printf(",\n \"merged_offset0\": {\"cube_new_merge\": %td, \"from_axes\": %td},\n \"loads\": {\n",
	       stored, want->offset0);
	for (int i = 2; i < argc; i++) verdict(argv[i], want, i == 2);
	printf("}}\n");
	return 0;
}
