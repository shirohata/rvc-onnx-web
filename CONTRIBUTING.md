# Contributing to RVC ONNX Converter

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/visgotti/rvc-onnx-web.git
   cd rvc-onnx-web
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Run tests**
   ```bash
   npm test
   ```

4. **Build**
   ```bash
   npm run build
   ```

## Code Style

- We use ESLint for linting
- Run `npm run lint` to check for issues
- Run `npm run lint:fix` to auto-fix issues

## Testing

- Tests are written using Vitest
- Run `npm test` to run all tests
- Run `npm run test:coverage` to generate coverage report
- Aim for >70% code coverage

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Reporting Bugs

Please open an issue with:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Model file details (if applicable, without sharing the actual file)

## Feature Requests

Open an issue with:
- Clear description of the feature
- Use case / motivation
- Any implementation ideas

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
