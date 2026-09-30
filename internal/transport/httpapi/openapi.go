package httpapi

// openAPISpec is the machine-readable contract served at /api/v1/openapi.json.
// It is written by hand rather than generated so the descriptions stay useful
// to a reader; docs/api.md is the narrative companion.
func openAPISpec(version string) map[string]any {
	tensor := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{"type": "string"},
			"dtype": map[string]any{"type": "string", "enum": []string{
				"float32", "float16", "bfloat16", "int32", "int64", "bool", "int8", "uint8"}},
			"shape": map[string]any{"type": "array", "items": map[string]any{"type": "integer"}},
		},
		"required": []string{"name", "dtype", "shape"},
	}

	errorSchema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"error": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"code":    map[string]any{"type": "string"},
					"message": map[string]any{"type": "string"},
				},
				"required": []string{"code", "message"},
			},
		},
		"required": []string{"error"},
	}

	errorResponse := func(desc string) map[string]any {
		return map[string]any{
			"description": desc,
			"content": map[string]any{
				"application/json": map[string]any{"schema": errorSchema},
			},
		}
	}

	questionSpec := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"type":         map[string]any{"type": "string", "enum": []string{"choice", "score", "noul"}},
			"instructions": map[string]any{"type": "string"},
			"criteria": map[string]any{
				"description": "choice: object of label -> description. score: array of " +
					"level descriptions. noul: omitted.",
				"oneOf": []any{
					map[string]any{"type": "object", "additionalProperties": map[string]any{"type": "string"}},
					map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
				},
			},
		},
		"required": []string{"type", "instructions"},
	}

	answer := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"type":          map[string]any{"type": "string"},
			"choice":        map[string]any{"type": "string"},
			"score":         map[string]any{"type": "number"},
			"noul":          map[string]any{"type": "number"},
			"legend":        map[string]any{"type": "object", "additionalProperties": map[string]any{"type": "string"}},
			"probabilities": map[string]any{"type": "object", "additionalProperties": map[string]any{"type": "number"}},
			"confidence":    map[string]any{"type": "number", "minimum": 0, "maximum": 1},
			"action": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"act_probability": map[string]any{"type": "number"},
				},
			},
		},
	}

	jsonBody := func(schema map[string]any) map[string]any {
		return map[string]any{
			"required": true,
			"content":  map[string]any{"application/json": map[string]any{"schema": schema}},
		}
	}
	jsonResp := func(desc string, schema map[string]any) map[string]any {
		return map[string]any{
			"description": desc,
			"content":     map[string]any{"application/json": map[string]any{"schema": schema}},
		}
	}

	return map[string]any{
		"openapi": "3.1.0",
		"info": map[string]any{
			"title":   "Laya Go Launcher API",
			"version": version,
			"description": "Typed decisions over any state, in one forward pass, served " +
				"by the laya model on TensorRT or ONNX Runtime. The same use cases back " +
				"the bundled GUI.",
		},
		"servers": []map[string]any{{"url": "http://127.0.0.1:8420"}},
		"paths": map[string]any{
			"/api/v1/health": map[string]any{
				"get": map[string]any{
					"summary":     "Liveness and runtime capability",
					"operationId": "getHealth",
					"responses": map[string]any{
						"200": jsonResp("Runtime status", map[string]any{
							"type": "object",
							"properties": map[string]any{
								"status":   map[string]any{"type": "string"},
								"version":  map[string]any{"type": "string"},
								"uptime_s": map[string]any{"type": "integer"},
								"kernel": map[string]any{
									"type": "object",
									"properties": map[string]any{
										"available": map[string]any{"type": "boolean"},
										"tensorrt":  map[string]any{"type": "string"},
										"error":     map[string]any{"type": "string"},
									},
								},
								"backend": map[string]any{
									"type": "object",
									"properties": map[string]any{
										"requested": map[string]any{"type": "string"},
										"selected":  map[string]any{"type": "string"},
										"note":      map[string]any{"type": "string"},
										"onnx":      map[string]any{"type": "object"},
									},
								},
								"device": map[string]any{"type": "object"},
								"engine": map[string]any{"type": "object"},
							},
						}),
					},
				},
			},
			"/api/v1/backends": map[string]any{
				"get": map[string]any{
					"summary":     "Which execution kernels this build and machine can offer",
					"operationId": "getBackends",
					"description": "Reports both kernels without loading a model, so a " +
						"client can choose before committing. 'selected' is what a load " +
						"would use right now.",
					"responses": map[string]any{
						"200": jsonResp("Kernel availability", map[string]any{
							"type": "object",
							"properties": map[string]any{
								"requested": map[string]any{"type": "string"},
								"selected":  map[string]any{"type": "string"},
								"note":      map[string]any{"type": "string"},
								"tensorrt":  map[string]any{"type": "object"},
								"onnx":      map[string]any{"type": "object"},
							},
						}),
					},
				},
			},
			"/api/v1/device": map[string]any{
				"get": map[string]any{
					"summary":     "Active device and VRAM",
					"operationId": "getDevice",
					"responses": map[string]any{
						"200": jsonResp("Device information", map[string]any{
							"type": "object",
							"properties": map[string]any{
								"name":               map[string]any{"type": "string"},
								"compute_capability": map[string]any{"type": "string"},
								"vram_free_mb":       map[string]any{"type": "integer"},
								"vram_total_mb":      map[string]any{"type": "integer"},
							},
						}),
					},
				},
			},
			"/api/v1/engine": map[string]any{
				"get": map[string]any{
					"summary":     "Describe the loaded model",
					"operationId": "getEngine",
					"responses": map[string]any{
						"200": jsonResp("Model description", map[string]any{
							"type": "object",
							"properties": map[string]any{
								"loaded":               map[string]any{"type": "boolean"},
								"path":                 map[string]any{"type": "string"},
								"backend":              map[string]any{"type": "string"},
								"runtime":              map[string]any{"type": "string"},
								"device":               map[string]any{"type": "string"},
								"contexts":             map[string]any{"type": "integer"},
								"activation_memory_mb": map[string]any{"type": "number"},
								"inputs":               map[string]any{"type": "array", "items": tensor},
								"outputs":              map[string]any{"type": "array", "items": tensor},
							},
						}),
						"404": errorResponse("No model is loaded"),
					},
				},
			},
			"/api/v1/engine/load": map[string]any{
				"post": map[string]any{
					"summary":     "Load a model, optionally switching kernel",
					"operationId": "loadEngine",
					"description": "A .engine plan is served by TensorRT and a .onnx graph " +
						"by ONNX Runtime; the format decides, not the extension alone. " +
						"Passing a different backend replaces the resident kernel in place.",
					"requestBody": jsonBody(map[string]any{
						"type": "object",
						"properties": map[string]any{
							"path": map[string]any{"type": "string"},
							"contexts": map[string]any{
								"type": "integer", "minimum": 0,
								"description": "0 or omitted picks a count from free VRAM (TensorRT only)",
							},
							"backend": map[string]any{
								"type": "string", "enum": []string{"auto", "tensorrt", "onnx"},
								"description": "Overrides the configured kernel for this load",
							},
							"provider": map[string]any{
								"type": "string", "enum": []string{"cuda", "cpu", "directml"},
								"description": "ONNX Runtime execution provider",
							},
						},
						"required": []string{"path"},
					}),
					"responses": map[string]any{
						"200": jsonResp("Model loaded", map[string]any{"type": "object"}),
						"400": errorResponse("Missing request, or an unknown backend name"),
						"404": errorResponse("Model file not found"),
						"409": errorResponse("Model incompatible with the selected kernel"),
						"503": errorResponse("The selected kernel's runtime is not installed"),
					},
				},
			},
			"/api/v1/engine/unload": map[string]any{
				"post": map[string]any{
					"summary":     "Release the model",
					"operationId": "unloadEngine",
					"responses": map[string]any{
						"200": jsonResp("Model released", map[string]any{
							"type":       "object",
							"properties": map[string]any{"unloaded": map[string]any{"type": "boolean"}},
						}),
					},
				},
			},
			"/api/v1/predict": map[string]any{
				"post": map[string]any{
					"summary":     "Evaluate typed questions over a state",
					"operationId": "predict",
					"description": "Every question is answered in one forward pass. " +
						"State may be a string, object, or array of turns.",
					"requestBody": jsonBody(map[string]any{
						"type": "object",
						"properties": map[string]any{
							"state": map[string]any{
								"oneOf": []any{
									map[string]any{"type": "string"},
									map[string]any{"type": "object"},
									map[string]any{"type": "array"},
								},
							},
							"questions": map[string]any{
								"type":                 "object",
								"additionalProperties": questionSpec,
							},
						},
						"required": []string{"state", "questions"},
					}),
					"responses": map[string]any{
						"200": jsonResp("Answers", map[string]any{
							"type": "object",
							"properties": map[string]any{
								"model": map[string]any{"type": "string"},
								"answers": map[string]any{
									"type":                 "object",
									"additionalProperties": answer,
								},
								"usage": map[string]any{
									"type": "object",
									"properties": map[string]any{
										"input_tokens":  map[string]any{"type": "integer"},
										"output_tokens": map[string]any{"type": "integer"},
									},
								},
								"timing": map[string]any{
									"type": "object",
									"properties": map[string]any{
										"total_ms":     map[string]any{"type": "number"},
										"tokenize_ms":  map[string]any{"type": "number"},
										"inference_ms": map[string]any{"type": "number"},
									},
								},
							},
						}),
						"400": errorResponse("Invalid question or schema"),
						"409": errorResponse("No engine is loaded"),
						"413": errorResponse("Request body too large"),
					},
				},
			},
			"/api/v1/tokenize": map[string]any{
				"post": map[string]any{
					"summary":     "Inspect tokenisation",
					"operationId": "tokenize",
					"requestBody": jsonBody(map[string]any{
						"type": "object",
						"properties": map[string]any{
							"text":          map[string]any{"type": "string"},
							"with_specials": map[string]any{"type": "boolean"},
						},
						"required": []string{"text"},
					}),
					"responses": map[string]any{
						"200": jsonResp("Tokenisation", map[string]any{
							"type": "object",
							"properties": map[string]any{
								"count":  map[string]any{"type": "integer"},
								"ids":    map[string]any{"type": "array", "items": map[string]any{"type": "integer"}},
								"tokens": map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
							},
						}),
					},
				},
			},
			"/api/v1/sequence": map[string]any{
				"post": map[string]any{
					"summary":     "Build the token sequence for one question",
					"operationId": "buildSequence",
					"requestBody": jsonBody(map[string]any{
						"type": "object",
						"properties": map[string]any{
							"state":    map[string]any{"type": "object"},
							"question": questionSpec,
						},
						"required": []string{"state", "question"},
					}),
					"responses": map[string]any{
						"200": jsonResp("Sequence layout", map[string]any{
							"type": "object",
							"properties": map[string]any{
								"ids":              map[string]any{"type": "array", "items": map[string]any{"type": "integer"}},
								"length":           map[string]any{"type": "integer"},
								"marker_positions": map[string]any{"type": "array", "items": map[string]any{"type": "integer"}},
								"options":          map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
								"truncated":        map[string]any{"type": "boolean"},
							},
						}),
					},
				},
			},
			"/api/v1/metrics": map[string]any{
				"get": map[string]any{
					"summary":     "Counters and latency percentiles",
					"operationId": "getMetrics",
					"responses": map[string]any{
						"200": jsonResp("Metrics", map[string]any{"type": "object"}),
					},
				},
			},
		},
	}
}
