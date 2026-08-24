# Automotive Software Engineering Prototype Knowledge Base

## Requirement quality

A software requirement should be specific, testable, unambiguous and traceable to verification evidence. Requirements should identify observable system behaviour and relevant constraints.

## Diagnostics

When a monitored input violates an allowed range, software may reject the value, preserve the previous valid state, create a diagnostic event, and expose the event to a monitoring component.

## Sensor validation

Sensor inputs should be validated before they are used by downstream control or decision logic. Validation can include range checks, plausibility checks, freshness checks and missing-value handling.

## Automated testing

A generated test should contain a clear precondition, input, expected result and pass/fail criterion. Boundary values and invalid inputs are important test cases for validation logic.

## Traceability

A useful AI-assisted engineering workflow should maintain links between the original requirement, retrieved evidence, generated implementation artifacts and verification tests.

## Human in the loop

For safety- or security-sensitive software, AI-generated artifacts should be reviewed and approved by a qualified engineer before integration or deployment.

## Security

Generated code should avoid unsafe assumptions about input data and should explicitly validate externally controlled inputs. Security-sensitive changes should undergo additional review and automated checks.
