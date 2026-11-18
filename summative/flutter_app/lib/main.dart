import 'package:flutter/material.dart';

import 'dart:convert';

import 'package:http/http.dart' as http;

void main() => runApp(const CostEstimatorApp());

const _navy = Color(0xFF0E1526);
const _gold = Color(0xFFC9A24B);
const _cream = Color(0xFFFAF7F2);

class CostEstimatorApp extends StatelessWidget {
  const CostEstimatorApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Hospital Cost Estimator',
      theme: ThemeData(
        scaffoldBackgroundColor: _cream,
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(seedColor: _navy),
      ),
      home: const PredictionPage(),
    );
  }
}

class InputField {
  final String key;
  final String label;
  final String placeholder;
  final String? suffix;
  final List<MapEntry<String, int>>? options;
  final TextEditingController controller = TextEditingController();
  int? selectedValue;
  InputField({
    required this.key,
    required this.label,
    required this.placeholder,
    this.suffix,
    this.options,
  });

  bool get isFilled => options == null
      ? controller.text.trim().isNotEmpty
      : selectedValue != null;
}

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});
  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  static const String apiUrl =
      "https://linear-regression-model-ry4e.onrender.com/predict";

  bool _isLoading = false;
  String? _resultText;
  bool _isError = false;

  // Real APR-MDC categories, codes 1-25
  static const _mdcOptions = [
    MapEntry("Nervous System", 1),
    MapEntry("Eye", 2),
    MapEntry("Ear, Nose, Mouth & Throat", 3),
    MapEntry("Respiratory System", 4),
    MapEntry("Circulatory System", 5),
    MapEntry("Digestive System", 6),
    MapEntry("Hepatobiliary System & Pancreas", 7),
    MapEntry("Musculoskeletal System & Connective Tissue", 8),
    MapEntry("Skin, Subcutaneous Tissue & Breast", 9),
    MapEntry("Endocrine, Nutritional & Metabolic", 10),
    MapEntry("Kidney & Urinary Tract", 11),
    MapEntry("Male Reproductive System", 12),
    MapEntry("Female Reproductive System", 13),
    MapEntry("Pregnancy, Childbirth & Puerperium", 14),
    MapEntry("Newborns & Other Neonates", 15),
    MapEntry("Blood & Immunological Disorders", 16),
    MapEntry("Myeloproliferative Diseases", 17),
    MapEntry("Infectious & Parasitic Diseases", 18),
    MapEntry("Mental Diseases & Disorders", 19),
    MapEntry("Alcohol/Drug Use Disorders", 20),
    MapEntry("Injuries & Poisonings", 21),
    MapEntry("Burns", 22),
    MapEntry("Factors Influencing Health Status", 23),
    MapEntry("Multiple Significant Trauma", 24),
    MapEntry("HIV Infections", 25),
  ];

  static const _severityOptions = [
    MapEntry("Minor", 1),
    MapEntry("Moderate", 2),
    MapEntry("Major", 3),
    MapEntry("Extreme", 4),
  ];
  static const _mortalityOptions = [
    MapEntry("Minor", 1),
    MapEntry("Moderate", 2),
    MapEntry("Major", 3),
    MapEntry("Extreme", 4),
  ];
  static const _careTypeOptions = [
    MapEntry("Medical", 0),
    MapEntry("Surgical", 1),
  ];
  static const _ageOptions = [
    MapEntry("0 to 17", 0),
    MapEntry("18 to 29", 1),
    MapEntry("30 to 49", 2),
    MapEntry("50 to 69", 3),
    MapEntry("70 or Older", 4),
  ];
  static const _admissionOptions = [
    MapEntry("Elective", 0),
    MapEntry("Urgent", 1),
    MapEntry("Emergency", 2),
    MapEntry("Newborn", 3),
    MapEntry("Trauma", 4),
    MapEntry("Not Available", 5),
  ];
  static const _edOptions = [MapEntry("No", 0), MapEntry("Yes", 1)];
  static const _paymentOptions = [
    MapEntry("Blue Cross/Blue Shield", 0),
    MapEntry("Department of Corrections", 1),
    MapEntry("Federal/State/Local/VA", 2),
    MapEntry("Managed Care, Unspecified", 3),
    MapEntry("Medicaid", 4),
    MapEntry("Medicare", 5),
    MapEntry("Miscellaneous/Other", 6),
    MapEntry("Private Health Insurance", 7),
    MapEntry("Self-Pay", 8),
    MapEntry("Unknown", 9),
  ];

  late final List<InputField> _fields = [
    InputField(
      key: "length_of_stay",
      label: "Length of stay",
      placeholder: "e.g. 3",
      suffix: "days",
    ),
    InputField(
      key: "apr_mdc_code",
      label: "Diagnostic category",
      placeholder: "Select category",
      options: _mdcOptions,
    ),
    InputField(
      key: "apr_medical_surgical_description",
      label: "Care type",
      placeholder: "Select care type",
      options: _careTypeOptions,
    ),
    InputField(
      key: "age_group",
      label: "Age group",
      placeholder: "Select age range",
      options: _ageOptions,
    ),
    InputField(
      key: "type_of_admission",
      label: "Admission type",
      placeholder: "Select admission",
      options: _admissionOptions,
    ),
    InputField(
      key: "emergency_department_indicator",
      label: "Admitted via ER?",
      placeholder: "Select an option",
      options: _edOptions,
    ),
    InputField(
      key: "apr_severity_of_illness_code",
      label: "Illness severity",
      placeholder: "Select severity",
      options: _severityOptions,
    ),
    InputField(
      key: "apr_risk_of_mortality",
      label: "Risk of mortality",
      placeholder: "Select risk level",
      options: _mortalityOptions,
    ),
    InputField(
      key: "payment_typology_1",
      label: "Insurance / payer",
      placeholder: "Select payer type",
      options: _paymentOptions,
    ),
  ];

  int get _filledCount => _fields.where((f) => f.isFilled).length;

  Future<void> _openPicker(InputField field) async {
    final chosen = await showModalBottomSheet<MapEntry<String, int>>(
      context: context,
      builder: (ctx) => SafeArea(
        child: ListView(
          shrinkWrap: true,
          children: field.options!.map((opt) {
            return ListTile(
              title: Text("${opt.value} - ${opt.key}"), // e.g. "2 - Emergency"
              onTap: () => Navigator.pop(ctx, opt),
            );
          }).toList(),
        ),
      ),
    );
    if (chosen != null) {
      setState(() {
        field.controller.text = "${chosen.value} - ${chosen.key}"; // selected field shows the same format
        field.selectedValue = chosen.value;
      });
    }
  }

  Future<void> _submitPrediction() async {
    final Map<String, dynamic> body = {};
    for (final field in _fields) {
      if (field.options != null) {
        if (field.selectedValue == null) {
          setState(() {
            _isError = true;
            _resultText = "Please select a value for '${field.label}'.";
          });
          return;
        }
        body[field.key] = field.selectedValue;
      } else {
        final text = field.controller.text.trim();
        if (text.isEmpty) {
          setState(() {
            _isError = true;
            _resultText = "Please fill in '${field.label}'.";
          });
          return;
        }
        final value = int.tryParse(text);
        if (value == null) {
          setState(() {
            _isError = true;
            _resultText = "'${field.label}' must be a whole number.";
          });
          return;
        }
        body[field.key] = value;
      }
    }

    setState(() {
      _isLoading = true;
      _resultText = null;
      _isError = false;
    });

    try {
      final response = await http.post(
        Uri.parse(apiUrl),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode(body),
      );
      final decoded = jsonDecode(response.body);

      if (response.statusCode == 200) {
        setState(() {
          _isError = false;
          _resultText =
              "Estimated Total Cost: \$${decoded['predicted_total_cost'].toStringAsFixed(2)}";
        });
      } else if (response.statusCode == 422) {
        final details = decoded['detail'] as List;
        final messages = details
            .map((d) {
              final field = (d['loc'] as List).last;
              return "$field: ${d['msg']}";
            })
            .join('\n');
        setState(() {
          _isError = true;
          _resultText = "Please check your input:\n$messages";
        });
      } else {
        setState(() {
          _isError = true;
          _resultText =
              "Server error (${response.statusCode}). Please try again.";
        });
      }
    } catch (e) {
      setState(() {
        _isError = true;
        _resultText =
            "Could not reach the server. Check your internet connection.";
      });
    } finally {
      setState(() => _isLoading = false);
    }
  }

  Widget _buildField(InputField field) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 14),
      child: TextField(
        controller: field.controller,
        readOnly: field.options != null,
        keyboardType: field.options == null
            ? TextInputType.number
            : TextInputType.none,
        onTap: field.options != null ? () => _openPicker(field) : null,
        onChanged: (_) => setState(() {}), // refreshes the X/9 progress count
        decoration: InputDecoration(
          labelText: field.label,
          hintText: field.placeholder,
          suffixText: field.suffix,
          suffixIcon: field.options != null
              ? const Icon(Icons.keyboard_arrow_down_rounded)
              : null,
          filled: true,
          fillColor: Colors.white,
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(10),
            borderSide: const BorderSide(color: Color(0xFFE0DCD3)),
          ),
        ),
      ),
    );
  }

  Widget _stepHeader(String step, String title, IconData icon) {
    return Padding(
      padding: const EdgeInsets.only(top: 24, bottom: 12),
      child: Row(
        children: [
          CircleAvatar(
            radius: 16,
            backgroundColor: _gold.withOpacity(0.15),
            child: Icon(icon, size: 16, color: _gold),
          ),
          const SizedBox(width: 10),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                step.toUpperCase(),
                style: TextStyle(
                  fontSize: 11,
                  color: _gold,
                  fontWeight: FontWeight.bold,
                  letterSpacing: 1,
                ),
              ),
              Text(
                title,
                style: const TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    for (final field in _fields) {
      field.controller.dispose();
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Container(
                color: _navy,
                padding: const EdgeInsets.symmetric(
                  vertical: 28,
                  horizontal: 20,
                ),
                child: RichText(
                  text: TextSpan(
                    style: const TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                    children: [
                      const TextSpan(text: "Know before you "),
                      TextSpan(
                        text: "go.",
                        style: TextStyle(color: _gold),
                      ),
                    ],
                  ),
                ),
              ),
              Padding(
                padding: const EdgeInsets.all(20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              "HOSPITAL COST ESTIMATOR",
                              style: TextStyle(
                                fontSize: 11,
                                color: _gold,
                                fontWeight: FontWeight.bold,
                                letterSpacing: 1,
                              ),
                            ),
                            const Text(
                              "Build your estimate",
                              style: TextStyle(
                                fontSize: 24,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ],
                        ),
                        Text(
                          "$_filledCount/9",
                          style: const TextStyle(color: Colors.grey),
                        ),
                      ],
                    ),
                    const Divider(height: 32),

                    _stepHeader("Step 01", "Visit details", Icons.add),
                    _buildField(_fields[0]), // length of stay
                    _buildField(_fields[1]), // mdc
                    _buildField(_fields[2]), // care type

                    _stepHeader(
                      "Step 02",
                      "Patient profile",
                      Icons.person_outline,
                    ),
                    _buildField(_fields[3]), // age
                    _buildField(_fields[4]), // admission type
                    _buildField(_fields[5]), // ER

                    _stepHeader(
                      "Step 03",
                      "Clinical & coverage",
                      Icons.favorite_border,
                    ),
                    _buildField(_fields[6]), // severity
                    _buildField(_fields[7]), // mortality
                    _buildField(_fields[8]), // payment

                    const SizedBox(height: 8),
                    SizedBox(
                      height: 52,
                      child: ElevatedButton.icon(
                        onPressed: _isLoading ? null : _submitPrediction,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: _navy,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(10),
                          ),
                        ),
                        icon: _isLoading
                            ? const SizedBox(
                                height: 16,
                                width: 16,
                                child: CircularProgressIndicator(
                                  strokeWidth: 2,
                                  color: Colors.white,
                                ),
                              )
                            : Icon(Icons.auto_awesome, color: _gold, size: 18),
                        label: const Text(
                          "Predict",
                          style: TextStyle(fontSize: 16),
                        ),
                      ),
                    ),
                    const SizedBox(height: 16),
                    if (_resultText != null)
                      Container(
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: _isError ? Colors.red[50] : Colors.green[50],
                          border: Border.all(
                            color: _isError ? Colors.red : Colors.green,
                          ),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Text(
                          _resultText!,
                          style: TextStyle(
                            color: _isError
                                ? Colors.red[900]
                                : Colors.green[900],
                            fontSize: 15,
                          ),
                        ),
                      ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
