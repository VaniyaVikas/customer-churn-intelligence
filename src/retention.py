
def recommend_retention(customer, churn_probability):
    recommendations = []

    contract = str(customer.get("Contract", "")).lower()
    tenure = customer.get("tenure", 0)
    support = str(customer.get("TechSupport", "")).lower()
    payment = str(customer.get("PaymentMethod", "")).lower()

    if churn_probability >= 0.70:
        risk = "High"
    elif churn_probability >= 0.40:
        risk = "Medium"
    else:
        risk = "Low"

    if risk == "High":
        recommendations.append(
            "Contact the customer and understand their concerns."
        )

    if "month-to-month" in contract:
        recommendations.append(
            "Offer a suitable long-term contract option."
        )

    if isinstance(tenure, (int, float)) and tenure <= 6:
        recommendations.append(
            "Provide onboarding help and check early customer satisfaction."
        )

    if support == "no":
        recommendations.append(
            "Offer a technical-support review if relevant."
        )

    if "electronic check" in payment:
        recommendations.append(
            "Explain available payment methods and their convenience."
        )

    if not recommendations:
        recommendations.append(
            "Continue regular customer satisfaction monitoring."
        )

    return {
        "risk_level": risk,
        "recommendations": recommendations,
    }
