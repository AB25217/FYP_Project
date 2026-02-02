package com.eventplanning.domain

import java.time.LocalDateTime
import java.time.Duration
import java.time.format.DateTimeFormatter

/** val (value) are used as once an event is created these properties
 * will be unable to change directly which protects the project keeping it more robust.
 */
data class Event(
    val id: String,
    val title: String,
    val dateTime: LocalDateTime,
    val venue: Venue,
    val description: String = "",
    val duration: Duration,
    val maxParticipants: Int = venue.capacity,
    private val registeredParticipants: MutableList<Participant> = mutableListOf()
) {
    companion object {
        private val DATE_TIME_FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm")
    }

    /**
     * The init block performs validation on the input parameters,
     * by checking conditions (require the id is not blank).
     */
    init {
        require(id.isNotBlank()) { "ID must not be blank." }
        require(title.isNotBlank()) { "Title must not be blank." }
        require(maxParticipants > 0) { "Max participants must be more than 0." }
        require(maxParticipants <= venue.capacity) {
            "Max participants ($maxParticipants) cannot exceed venue capacity (${venue.capacity})."
        }
        require(!duration.isNegative && !duration.isZero) {
            "Duration must be positive."
        }
    }

    /**
     * Returns a read-only copy of the registered participants list.
     * This prevents external modification of the internal state.
     */
    fun getRegisteredParticipants(): List<Participant> = registeredParticipants.toList()

    /**
     * Whilst registering a participant, checks if statements instead of a true/false
     * allows end users to know why their registration failed
     */
    fun registerParticipant(participant: Participant): RegistrationResult {
        if (isFull()) {
            return RegistrationResult.EventFull
        }

        if (isParticipantRegistered(participant)) {
            return RegistrationResult.AlreadyRegistered
        }

        registeredParticipants.add(participant)
        return RegistrationResult.Success
    }

    /**
     * Unregisters a participant from the event.
     * @return true if participant was found and removed, false otherwise
     */
    fun unregisterParticipant(participant: Participant): Boolean {
        return registeredParticipants.removeIf { it.id == participant.id }
    }

    /**
     * Checks if the event has reached maximum capacity.
     */
    fun isFull(): Boolean = registeredParticipants.size >= maxParticipants

    /**
     * Returns the current number of registered participants.
     */
    fun getCurrentCapacity(): Int = registeredParticipants.size

    /**
     * Returns the number of available spots remaining.
     */
    fun getAvailableSpots(): Int = maxParticipants - registeredParticipants.size

    /**
     * Checks if a specific participant is already registered.
     */
    fun isParticipantRegistered(participant: Participant): Boolean =
        registeredParticipants.any { it.id == participant.id }

    /**
     * Calculates and returns the event's end time.
     */
    fun getEndTime(): LocalDateTime = dateTime.plus(duration)

    /**
     * Checks if this event conflicts with another event (same venue, overlapping time).
     * uses StartA < EndB && EndA > StartB logic to determine overlap.
     */
    fun conflictsWith(other: Event): Boolean {
        if (this.id == other.id) {
            return false
        }

        if (this.venue.id != other.venue.id) {
            return false
        }
        val thisEnd = this.getEndTime()
        val otherEnd = other.getEndTime()

        return this.dateTime < otherEnd && thisEnd > other.dateTime
    }

    override fun toString(): String {
        return "$title @ ${venue.name} on ${dateTime.format(DATE_TIME_FORMATTER)} " +
                "(${getCurrentCapacity()}/$maxParticipants registered)"
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is Event) return false
        return id == other.id
    }

    override fun hashCode(): Int = id.hashCode()
}

/**
 * Sealed class representing the result of a registration attempt.
 * Provides type-safe error handling. by defining specific a strict set of possible outcomes.
 */
sealed class RegistrationResult {
    object Success : RegistrationResult()
    object EventFull : RegistrationResult()
    object AlreadyRegistered : RegistrationResult()
}