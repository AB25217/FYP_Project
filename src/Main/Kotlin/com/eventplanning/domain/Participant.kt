package com.eventplanning.domain

data class Participant(
    val id: String,
    val name: String,
    val email: String,
    val phone: String = "",
    val organization: String = ""
) {
    /**
     * Regex pattern to validate email format.
     * makes sure the email contains an '@' symbol and a domain.
     */
    companion object {
        private val EMAIL_REGEX = Regex("^[A-Za-z0-9+_.-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$")
    }

    /**
     * this will run every time an instance of Participant is created
     * it enforces some basic validation rules on the properties using require statements.
     */
    init {
        require(id.isNotBlank()) { "ID must not be blank." }
        require(name.isNotBlank()) { "Name must not be blank." }
        require(email.matches(EMAIL_REGEX)) {
            "Invalid email format. Expected format: example@domain.com"
        }
    }

    /**
     * returns a formatted string combining the participant's name and email.
     */
    fun getDisplayName(): String = "$name ($email)"

    /**
     * Overrides the default toString method to return the participant's display name.
     */
    override fun toString(): String = getDisplayName()
}