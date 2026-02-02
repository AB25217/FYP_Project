package com.eventplanning.domain

data class Venue(
    val id: String,
    val name: String,
    val capacity: Int,
    val location: String,
    val address: String,
    val facilities: List<String> = emptyList()
) {
    /**
     * The init blocks run every time an instance of Venue is created
     * It enforces some basic validation rules on the properties using require statements.
     */
    init {
        require(id.isNotBlank()) { "ID must not be empty." }
        require(name.isNotBlank()) { "Name must not be empty." }
        require(capacity > 0) { "Capacity must be greater than zero." }
        require(location.isNotBlank()) { "Location must not be empty." }
        require(address.isNotBlank()) { "Address must not be empty." }
        require(facilities.all { it.isNotBlank() }) { "Facility names must not be empty." }
    }
    /**
     * Returns a formatted string of facilities or "None" if the list is empty.
     */
    fun getFacilitiesDisplay(): String =
        if (facilities.isEmpty()) "None" else facilities.joinToString(", ")

    /**
     * Overrides the default toString method to return the venue's name and capacity.
     */
    override fun toString(): String = "$name (Capacity: $capacity)"
}