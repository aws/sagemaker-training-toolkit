// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the 'License'). You
// may not use this file except in compliance with the License. A copy of
// the License is located at
//
//     http://aws.amazon.com/apache2.0/
//
// or in the 'license' file accompanying this file. This file is
// distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
// ANY KIND, either express or implied. See the License for the specific
// language governing permissions and limitations under the License.

#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include "jsmn.h"

static int jsoneq(const char *json, jsmntok_t *tok, const char *s)
{
    if (tok->type == JSMN_STRING && (int) strlen(s) == tok->end - tok->start &&
        strncmp(json + tok->start, s, tok->end - tok->start) == 0)
    {
        return 0;
    }
    return -1;
}

int gethostname(char *name, size_t len)
{
    int r;
    FILE *file = fopen("/opt/ml/input/config/resourceconfig.json", "r");

    fseek(file, 0, SEEK_END);

    long length = ftell(file);
    fseek(file, 0, SEEK_SET);

    char *json_string = malloc(length + 1);
    fread(json_string, 1, length, file);
    fclose(file);
    json_string[length] = '\0';

    jsmn_parser parser;
    jsmntok_t token[1024];


    jsmn_init(&parser);
    r = jsmn_parse(&parser, json_string, strlen(json_string), token, sizeof(token) / sizeof(token[0]));


    if (r < 0) {
        printf("Failed to parse JSON: %d\n", r);
        return 1;
    }

    /* Assume the top-level element is an object */
    if (r < 1 || token[0].type != JSMN_OBJECT) {
        printf("Object expected\n");
        return 1;
    }

    /* Loop over all keys of the root object */
    int i;
    for (i = 1; i < r; i++)
    {
        if (jsoneq(json_string, &token[i], "current_host") == 0)
        {
            // strndup guarantees that val is null terminated. See https://linux.die.net/man/3/strndup
            char *val = strndup(json_string + token[i + 1].start, token[i + 1].end - token[i + 1].start);

            // Copy val into name. If strlen(val) > strlen(name) only len characters are copied
            strncpy(name, val, len);

            // As per posix (http://man7.org/linux/man-pages/man2/gethostname.2.html),
            // len is the size of the buffer, so we null terminate the last
            // position in the buffer
            name[len - 1] = '\0';

            free(val);
            free(json_string);
            return 0;
        }
    }

    free(json_string);
    return 1;
}

static PyObject *gethostname_call(PyObject *self, PyObject *args)
{
    long unsigned command;
    char name[40];

    if (!PyArg_ParseTuple(args, "k", &command))
    {
        return NULL;
    }

    gethostname(name, command);

    return Py_BuildValue("s", name);
}

static PyMethodDef GetHostnameMethods[] = {
    {
        "call",
        gethostname_call,
        METH_VARARGS,
    },
    {NULL, NULL, 0, NULL}, // sentinel
};

#if PY_MAJOR_VERSION >= 3
static PyModuleDef gethostnamemodule = {
    PyModuleDef_HEAD_INIT,
    "gethostname",
    "Returns the current host defined in resourceconfig.json",
    -1,
    GetHostnameMethods,
};

PyMODINIT_FUNC PyInit_gethostname()
{
    return PyModule_Create(&gethostnamemodule);
}
#else
PyMODINIT_FUNC initgethostname()
{
    PyObject *module;

    module = Py_InitModule3(
        "gethostname", GetHostnameMethods, "Returns the current host defined in resourceconfig.json");
}
#endif
