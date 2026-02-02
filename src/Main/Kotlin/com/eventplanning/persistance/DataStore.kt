package com.eventplanning.persistance

import com.eventplanning.domain.*
import java.sql.Connection
import java.sql.DriverManager
import java.sql.SQLException
import java.time.LocalDateTime
import java.time.Duration
import java.time.format.DateTimeFormatter

/**
 * DataStore class implements the Repository interface to manage
 * the persistence of event planning data using SQLite.
 */
class DataStore(private val dbPath: String = "events.db") : Repository {

    /**
     * Sql object containing all SQL queries used in the DataStore.
     *
     * create_venues, create_participants, create_events, create_registrations
     * are table definitions for the database schema.
     */
    private object SqlQueries {
        const val CREATE_VENUES = """
            CREATE TABLE IF NOT EXISTS Venues (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                capacity INTEGER NOT NULL,
                location TEXT NOT NULL,
                address TEXT NOT NULL,
                facilities TEXT
            )"""

        const val CREATE_PARTICIPANTS = """
            CREATE TABLE IF NOT EXISTS Participants (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                phone TEXT,
                organization TEXT
            )"""

        const val CREATE_EVENTS = """
            CREATE TABLE IF NOT EXISTS Events (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                dateTime TEXT NOT NULL,
                venue_id TEXT NOT NULL,
                description TEXT,
                duration_minutes INTEGER NOT NULL,
                max_participants INTEGER NOT NULL,
                FOREIGN KEY (venue_id) REFERENCES Venues(id)
            )"""

        const val CREATE_REGISTRATIONS = """
            CREATE TABLE IF NOT EXISTS Event_Registrations (
                event_id TEXT,
                participant_id TEXT,
                registration_date TEXT NOT NULL,
                PRIMARY KEY (event_id, participant_id),
                FOREIGN KEY (event_id) REFERENCES Events(id) ON DELETE CASCADE,
                FOREIGN KEY (participant_id) REFERENCES Participants(id) ON DELETE CASCADE
            )"""

        /**
         *SQL queries for operations on Venues, Participants, Events, and Registrations.
         * these strings describe what to ask the database to do.
         */
        const val INSERT_VENUE =
            "INSERT OR REPLACE INTO Venues (id, name, capacity, location, address, facilities) VALUES (?, ?, ?, ?, ?, ?)"
        const val SELECT_VENUES = "SELECT * FROM Venues"
        const val DELETE_VENUE = "DELETE FROM Venues WHERE id = ?"

        const val INSERT_PARTICIPANT =
            "INSERT OR REPLACE INTO Participants (id, name, email, phone, organization) VALUES (?, ?, ?, ?, ?)"
        const val SELECT_PARTICIPANTS = "SELECT * FROM Participants"
        const val DELETE_PARTICIPANT = "DELETE FROM Participants WHERE id = ?"

        const val INSERT_EVENT =
            "INSERT OR REPLACE INTO Events (id, title, dateTime, venue_id, description, duration_minutes, max_participants) VALUES (?, ?, ?, ?, ?, ?, ? )"
        const val SELECT_EVENTS = "SELECT * FROM Events"
        const val DELETE_EVENT = "DELETE FROM Events WHERE id = ?"

        const val DELETE_REGISTRATIONS = "DELETE FROM Event_Registrations WHERE event_id = ?"
        const val INSERT_REGISTRATION =
            "INSERT INTO Event_Registrations (event_id, participant_id, registration_date) VALUES (?, ?, ?)"
        const val SELECT_ALL_REGISTRATIONS = "SELECT event_id, participant_id FROM Event_Registrations"
    }

    /**
     *connection is a handle to the SQLite database.
     * dateTimeFormatter is used to format date-time strings consistently.
     */
    private var connection: Connection? = null
    private val dateTimeFormatter = DateTimeFormatter.ISO_LOCAL_DATE_TIME

    /**
     * Establishes a connection to the SQLite database loading sqlite JDBC driver.
     * opens the database file specified by dbPath.
     * enables foreign key constraints to maintain referential integrity.
     * handles SQLExceptions to provide meaningful error messages for connection issues.
     * prints success message upon successful connection.
     */
    override fun connect() {
        try {
            Class.forName("org.sqlite.JDBC")
            connection = DriverManager.getConnection("jdbc:sqlite:$dbPath")

            connection?.createStatement()?.use { stmt ->
                stmt.execute("PRAGMA foreign_keys = ON;")
            }
            println("Database connected successfully: $dbPath")
        } catch (e: SQLException) {
            System.err.println("Database Error: Could not open file at '$dbPath'. Check permissions or path validity.")
            throw IllegalArgumentException("Invalid database path or permissions issue: $dbPath", e)
        } catch (e: Exception) {
            System.err.println("Error connecting to database: ${e.message}")
            throw e
        }
    }

    /**
     * requires connection to be established, if not throws IllegalStateException.
     * creates the necessary tables (Venues, Participants, Events, Event_Registrations)
     * if they do not already exist.
     */
    override fun initializeStorage() {
        val conn = connection ?: throw IllegalStateException("Database not connected")
        try {
            conn.createStatement().use { stmt ->
                stmt.executeUpdate(SqlQueries.CREATE_VENUES)
                stmt.executeUpdate(SqlQueries.CREATE_PARTICIPANTS)
                stmt.executeUpdate(SqlQueries.CREATE_EVENTS)
                stmt.executeUpdate(SqlQueries.CREATE_REGISTRATIONS)
            }
            println("Database schema initialized successfully.")
        } catch (e: SQLException) {
            System.err.println("Error creating tables: ${e.message}")
            throw e
        }
    }

    /**
     * takes a Venue object and saves it to the database.
     * if a row with the same ID exists, it updates that row.
     * otherwise, it inserts a new row.
     * facilities.joinToString(",") converts the list into a comma-separated string.
     * synchronized to ensure thread safety during database write operations.
     */
    @Synchronized
    override fun saveVenue(venue: Venue): Boolean {
        val conn = connection ?: return false
        return try {
            conn.prepareStatement(SqlQueries.INSERT_VENUE).use { stmt ->
                stmt.setString(1, venue.id)
                stmt.setString(2, venue.name)
                stmt.setInt(3, venue.capacity)
                stmt.setString(4, venue.location)
                stmt.setString(5, venue.address)
                stmt.setString(6, venue.facilities.joinToString(","))
                stmt.executeUpdate()
                true
            }
        } catch (e: SQLException) {
            System.err.println("Error saving venue: ${e.message}")
            false
        }
    }

    /**
     * deletes a venue from the database based on its ID.
     * returns true if a row was deleted, false otherwise.
     */
    @Synchronized
    override fun deleteVenue(id: String): Boolean {
        val conn = connection ?: return false
        return try {
            conn.prepareStatement(SqlQueries.DELETE_VENUE).use { stmt ->
                stmt.setString(1, id)
                stmt.executeUpdate() > 0
            }
        } catch (e: SQLException) {
            System.err.println("Error deleting venue: ${e.message}")
            false
        }
    }

    /**
     * retrieves all venues from the database.
     * loops through the result set and constructs Venue objects.
     * column labels made such as "id", "name", "capacity", etc.
     * facilities string is split at commas, trimmed, and filtered empty parts to create a list.
     *
     */
    override fun loadAllVenues(): List<Venue> {
        val conn = connection ?: return emptyList()
        val venues = mutableListOf<Venue>()
        try {
            conn.createStatement().use { stmt ->
                val rs = stmt.executeQuery(SqlQueries.SELECT_VENUES)
                while (rs.next()) {
                    venues.add(
                        Venue(
                            id = rs.getString("id"),
                            name = rs.getString("name"),
                            capacity = rs.getInt("capacity"),
                            location = rs.getString("location"),
                            address = rs.getString("address"),
                            facilities = rs.getString("facilities")
                                ?.split(",")
                                ?.map { it.trim() }
                                ?.filter { it.isNotBlank() }
                                ?: emptyList()
                        )
                    )
                }
            }
        } catch (e: SQLException) {
            System.err.println("Error loading venues: ${e.message}")
        }
        return venues
    }

    /**
     * Similar to saveVenue, but for Participant objects.
     */
    @Synchronized
    override fun saveParticipant(participant: Participant): Boolean {
        val conn = connection ?: return false
        return try {
            conn.prepareStatement(SqlQueries.INSERT_PARTICIPANT).use { stmt ->
                stmt.setString(1, participant.id)
                stmt.setString(2, participant.name)
                stmt.setString(3, participant.email)
                stmt.setString(4, participant.phone)
                stmt.setString(5, participant.organization)
                stmt.executeUpdate()
                true
            }
        } catch (e: SQLException) {
            System.err.println("Error saving participant: ${e.message}")
            false
        }
    }
    /**
     * Similar to deleteVenue, but for Participant objects.
     */
    @Synchronized
    override fun deleteParticipant(id: String): Boolean {
        val conn = connection ?: return false
        return try {
            conn.prepareStatement(SqlQueries.DELETE_PARTICIPANT).use { stmt ->
                stmt.setString(1, id)
                stmt.executeUpdate() > 0
            }
        } catch (e: SQLException) {
            System.err.println("Error deleting participant: ${e.message}")
            false
        }
    }
    /**
     * Similar to loadAllVenues, but for Participant objects.
     */
    override fun loadAllParticipants(): List<Participant> {
        val conn = connection ?: return emptyList()
        val participants = mutableListOf<Participant>()
        try {
            conn.createStatement().use { stmt ->
                val rs = stmt.executeQuery(SqlQueries.SELECT_PARTICIPANTS)
                while (rs.next()) {
                    participants.add(
                        Participant(
                            id = rs.getString("id"),
                            name = rs.getString("name"),
                            email = rs.getString("email"),
                            phone = rs.getString("phone") ?: "",
                            organization = rs.getString("organization") ?: ""
                        )
                    )
                }
            }
        } catch (e: SQLException) {
            System.err.println("Error loading participants: ${e.message}")
        }
        return participants
    }

    /**
     * saves an Event object to the database.
     * uses a transaction to ensure both event and its registrations are saved atomically.
     * turns off auto-commit mode to start the transaction.
     * updates every row in Event_Registrations for the event.
     * deletes old registrations and inserts current ones.
     * commits the transaction if all operations succeed.
     */
    @Synchronized
    override fun saveEvent(event: Event): Boolean {
        val conn = connection ?: return false
        val originalAutoCommit = conn.autoCommit

        return try {
            conn.autoCommit = false

            conn.prepareStatement(SqlQueries.INSERT_EVENT).use { stmt ->
                stmt.setString(1, event.id)
                stmt.setString(2, event.title)
                stmt.setString(3, event.dateTime.format(dateTimeFormatter))
                stmt.setString(4, event.venue.id)
                stmt.setString(5, event.description)
                stmt.setLong(6, event.duration.toMinutes())
                stmt.setInt(7, event.maxParticipants)
                stmt.executeUpdate()
            }

            conn.prepareStatement(SqlQueries.DELETE_REGISTRATIONS).use { stmt ->
                stmt.setString(1, event.id)
                stmt.executeUpdate()
            }

            conn.prepareStatement(SqlQueries.INSERT_REGISTRATION).use { stmt ->
                for (participant in event.getRegisteredParticipants()) {
                    stmt.setString(1, event.id)
                    stmt.setString(2, participant.id)
                    stmt.setString(3, LocalDateTime.now().format(dateTimeFormatter))
                    stmt.executeUpdate()
                }
            }

            conn.commit()
            true
        } catch (e: SQLException) {
            try {
                conn.rollback()
            } catch (rollbackEx: SQLException) {
                System.err.println("Rollback failed: ${rollbackEx.message}")
            }
            System.err.println("Error saving event: ${e.message}")
            false
        } finally {
            conn.autoCommit = originalAutoCommit
        }
    }

    /**
     * deletes an event from the database based on its ID.
     * due to foreign key constraints with ON DELETE CASCADE,
     * associated registrations are automatically deleted.
     */
    @Synchronized
    override fun deleteEvent(id: String): Boolean {
        val conn = connection ?: return false
        return try {
            conn.prepareStatement(SqlQueries.DELETE_EVENT).use { stmt ->
                stmt.setString(1, id)
                stmt.executeUpdate() > 0
            }
        } catch (e: SQLException) {
            System.err.println("Error deleting event: ${e.message}")
            false
        }
    }

    /**
     * opens the database connection and retrieves all events.
     */
    override fun loadAllEvents(
        venueLookup: (String) -> Venue?,
        participantLookup: (String) -> Participant?
    ): List<Event> {
        val conn = connection ?: return emptyList()
        val events = mutableListOf<Event>()

        val registrationMap = HashMap<String, MutableList<String>>()

        /**
         *reads all rows from the Event_Registrations table.
         *builds a map of event IDs to lists of participant IDs.
         */
        try {
            conn.createStatement().use { stmt ->
                val rs = stmt.executeQuery(SqlQueries.SELECT_ALL_REGISTRATIONS)
                while (rs.next()) {
                    val eventId = rs.getString("event_id")
                    val participantId = rs.getString("participant_id")
                    registrationMap.computeIfAbsent(eventId) { mutableListOf() }.add(participantId)
                }
            }
        } catch (e: SQLException) {
            System.err.println("Error pre-loading registrations: ${e.message}")
        }
        /**
         * for each event row
         * gets id, venue_id, max_participants
         * asks venueLookup for the Venue object by id
         */
        try {
            conn.createStatement().use { stmt ->
                val rs = stmt.executeQuery(SqlQueries.SELECT_EVENTS)
                while (rs.next()) {
                    val eventId = rs.getString("id")
                    val venueId = rs.getString("venue_id")
                    val maxParticipants = rs.getInt("max_participants")
                    var venue = venueLookup(venueId)

                    /**
                     * in the case a venue is deleted from database but events still reference it
                     * logs a warning and creates a placeholder Venue object with default values.
                     */
                    if (venue == null) {
                        System.err.println("Warning: Event '$eventId' references missing venue '$venueId'. Loading as placeholder.")
                        venue = Venue(
                            id = "unknown_venue",
                            name = "UNKNOWN VENUE (Missing Data)",
                            capacity = maxParticipants,
                            location = "Unknown",
                            address = "Unknown"
                        )
                    }
                    /**
                     * construct a proper domain Event from database row.
                     * parses dateTime string into LocalDateTime object.
                     * duration is converted from minutes to Duration object.
                     */
                    val event = Event(
                        id = eventId,
                        title = rs.getString("title") + (if (venue.id == "unknown_venue") " [DATA ERROR]" else ""),
                        dateTime = LocalDateTime.parse(rs.getString("dateTime"), dateTimeFormatter),
                        venue = venue,
                        description = rs.getString("description") ?: "",
                        duration = Duration.ofMinutes(rs.getLong("duration_minutes")),
                        maxParticipants = maxParticipants
                    )
                    /**
                     * looks up each participant ID registered for this event.
                     * uses participantLookup to get the Participant object.
                     * registers the participant with the event.
                     */
                    registrationMap[eventId]?.forEach { participantId ->
                        participantLookup(participantId)?.let { participant ->
                            event.registerParticipant(participant)
                        }
                    }

                    events.add(event)
                }
            }
        } catch (e: SQLException) {
            System.err.println("Error loading events: ${e.message}")
        }
        return events
    }

    /**
     * close the database connection when application is shutting down.
     */
    override fun disconnect() {
        try {
            connection?.close()
            println("Database disconnected.")
        } catch (e: SQLException) {
            System.err.println("Error disconnecting: ${e.message}")
        }
    }
}