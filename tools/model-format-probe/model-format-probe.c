#include <dirent.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define SAFETENSORS_HEADER_MAX (64ULL * 1024ULL * 1024ULL)
#define MAX_SCAN_DEPTH 5

struct result {
    const char *format;
    const char *confidence;
    char source[PATH_MAX];
    int found;
    int conflict;
};

static uint64_t read_le64(const unsigned char b[8]) {
    uint64_t value = 0;
    for (int i = 7; i >= 0; --i) {
        value = (value << 8) | (uint64_t)b[i];
    }
    return value;
}

static int file_size(FILE *fp, uint64_t *size_out) {
    long current = ftell(fp);
    if (current < 0) {
        return -1;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        return -1;
    }
    long end = ftell(fp);
    if (end < 0) {
        return -1;
    }
    if (fseek(fp, current, SEEK_SET) != 0) {
        return -1;
    }
    *size_out = (uint64_t)end;
    return 0;
}

static int contains_token(const char *text, const char *token) {
    return strstr(text, token) != NULL;
}

static int probe_safetensors(FILE *fp, const unsigned char prefix[16]) {
    uint64_t size = 0;
    uint64_t header_len = read_le64(prefix);
    char *header = NULL;
    size_t got;
    int ok = 0;

    if (header_len < 2 || header_len > SAFETENSORS_HEADER_MAX) {
        return 0;
    }
    if (file_size(fp, &size) != 0 || size < 8 || header_len > size - 8) {
        return 0;
    }
    header = (char *)malloc((size_t)header_len + 1);
    if (header == NULL) {
        return 0;
    }
    if (fseek(fp, 8, SEEK_SET) != 0) {
        free(header);
        return 0;
    }
    got = fread(header, 1, (size_t)header_len, fp);
    if (got != (size_t)header_len) {
        free(header);
        return 0;
    }
    header[header_len] = '\0';

    ok = header[0] == '{' &&
         contains_token(header, "\"dtype\"") &&
         contains_token(header, "\"shape\"") &&
         contains_token(header, "\"data_offsets\"");

    free(header);
    return ok;
}

static int probe_file(const char *path, const char **format, const char **confidence) {
    FILE *fp;
    unsigned char prefix[16];
    size_t got;

    *format = NULL;
    *confidence = NULL;

    fp = fopen(path, "rb");
    if (fp == NULL) {
        return 0;
    }
    memset(prefix, 0, sizeof(prefix));
    got = fread(prefix, 1, sizeof(prefix), fp);
    if (got < 4) {
        fclose(fp);
        return 0;
    }

    if (memcmp(prefix, "GGUF", 4) == 0) {
        *format = "gguf";
        *confidence = "high";
        fclose(fp);
        return 1;
    }

    if (got >= 10 && probe_safetensors(fp, prefix)) {
        *format = "hf_safetensors";
        *confidence = "high";
        fclose(fp);
        return 1;
    }

    if (got >= 4 && prefix[0] == 'P' && prefix[1] == 'K' &&
        prefix[2] == 0x03 && prefix[3] == 0x04) {
        *format = "hf_torch_legacy";
        *confidence = "medium";
        fclose(fp);
        return 1;
    }

    if (got >= 2 && prefix[0] == 0x80 &&
        (prefix[1] == 0x02 || prefix[1] == 0x03 ||
         prefix[1] == 0x04 || prefix[1] == 0x05)) {
        *format = "hf_torch_legacy";
        *confidence = "low";
        fclose(fp);
        return 1;
    }

    fclose(fp);
    return 0;
}

static void record_result(struct result *res, const char *format,
                          const char *confidence, const char *source) {
    if (!res->found) {
        res->format = format;
        res->confidence = confidence;
        snprintf(res->source, sizeof(res->source), "%s", source);
        res->found = 1;
        return;
    }
    if (strcmp(res->format, format) != 0) {
        res->conflict = 1;
    }
}

static void scan_path(const char *path, int depth, struct result *res) {
    struct stat st;
    const char *format;
    const char *confidence;

    if (res->conflict || stat(path, &st) != 0) {
        return;
    }

    if (S_ISREG(st.st_mode)) {
        if (probe_file(path, &format, &confidence)) {
            record_result(res, format, confidence, path);
        }
        return;
    }

    if (!S_ISDIR(st.st_mode) || depth >= MAX_SCAN_DEPTH) {
        return;
    }

    DIR *dir = opendir(path);
    if (dir == NULL) {
        return;
    }

    for (;;) {
        struct dirent *entry = readdir(dir);
        char child[PATH_MAX];
        int written;

        if (entry == NULL || res->conflict) {
            break;
        }
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        written = snprintf(child, sizeof(child), "%s/%s", path, entry->d_name);
        if (written < 0 || (size_t)written >= sizeof(child)) {
            continue;
        }
        scan_path(child, depth + 1, res);
    }

    closedir(dir);
}

int main(int argc, char **argv) {
    struct result res;
    struct stat st;

    if (argc != 2) {
        fprintf(stderr, "usage: %s <model-file-or-directory>\n", argv[0]);
        return 2;
    }

    if (stat(argv[1], &st) != 0) {
        printf("model_format=unknown\n");
        printf("probe_source=%s\n", argv[1]);
        printf("confidence=none\n");
        fprintf(stderr, "error: cannot stat %s: %s\n", argv[1], strerror(errno));
        return 2;
    }

    memset(&res, 0, sizeof(res));
    scan_path(argv[1], 0, &res);

    if (res.conflict) {
        printf("model_format=unknown\n");
        printf("probe_source=%s\n", argv[1]);
        printf("confidence=conflict\n");
        fprintf(stderr, "error: multiple model formats detected under %s\n", argv[1]);
        return 2;
    }

    if (!res.found) {
        printf("model_format=unknown\n");
        printf("probe_source=%s\n", argv[1]);
        printf("confidence=none\n");
        return 1;
    }

    printf("model_format=%s\n", res.format);
    printf("probe_source=%s\n", res.source);
    printf("confidence=%s\n", res.confidence);
    printf("probe_type=binary\n");
    return 0;
}
